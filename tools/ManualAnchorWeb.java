import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.awt.Desktop;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.Executors;

public class ManualAnchorWeb {
    static final class Item {
        int promptIndex;
        String baseUnit;
        String context;
        String syllable;
        String className;
        Path wav;
        int occurrence;
        double prevCueMs;
        double cueMs;
        double nextCueMs;

        String key() { return promptIndex + ":" + occurrence; }
    }

    static final class Label {
        double ms;
        String status;
        Label(double ms, String status) { this.ms = ms; this.status = status; }
    }

    private final List<Item> items;
    private final Path labelsPath;
    private final Map<String, Label> labels = Collections.synchronizedMap(new LinkedHashMap<>());
    private final Map<Integer, Path> wavByPrompt = new HashMap<>();

    ManualAnchorWeb(Path manifest, Path labelsPath) throws IOException {
        this.items = readManifest(manifest);
        this.labelsPath = labelsPath;
        for (Item item : items) wavByPrompt.put(item.promptIndex, item.wav);
        loadLabels();
    }

    private static List<Item> readManifest(Path path) throws IOException {
        List<Item> result = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.split("\\t", -1);
            if (p.length < 10) throw new IOException("Bad manifest row: " + line);
            Item item = new Item();
            item.promptIndex = Integer.parseInt(p[0]);
            item.baseUnit = p[1];
            item.context = p[2];
            item.syllable = p[3];
            item.className = p[4];
            item.wav = Paths.get(p[5]);
            item.occurrence = Integer.parseInt(p[6]);
            item.prevCueMs = Double.parseDouble(p[7]);
            item.cueMs = Double.parseDouble(p[8]);
            item.nextCueMs = Double.parseDouble(p[9]);
            result.add(item);
        }
        return result;
    }

    private void loadLabels() throws IOException {
        if (!Files.isRegularFile(labelsPath)) return;
        for (String line : Files.readAllLines(labelsPath, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            String[] p = line.split("\\t", -1);
            if (p.length < 4) continue;
            try {
                labels.put(p[0] + ":" + p[1], new Label(Double.parseDouble(p[2]), p[3]));
            } catch (NumberFormatException ignored) {}
        }
    }

    private synchronized void saveLabels() throws IOException {
        Files.createDirectories(labelsPath.getParent());
        List<String> out = new ArrayList<>();
        out.add("# prompt_index\toccurrence\tanchor_ms_after_cue\tstatus");
        for (Item item : items) {
            Label label = labels.get(item.key());
            if (label != null) {
                out.add(String.format(Locale.US, "%d\t%d\t%.3f\t%s",
                        item.promptIndex, item.occurrence, label.ms, label.status));
            }
        }
        Path temp = labelsPath.resolveSibling(labelsPath.getFileName() + ".tmp");
        Files.write(temp, out, StandardCharsets.UTF_8);
        try {
            Files.move(temp, labelsPath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ex) {
            Files.move(temp, labelsPath, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private String manifestTsv() {
        StringBuilder b = new StringBuilder();
        b.append("prompt_index\tbase_unit\tcontext\tsyllable\tclass_name\toccurrence\tprev_cue_ms\tcue_ms\tnext_cue_ms\n");
        for (Item i : items) {
            b.append(i.promptIndex).append('\t').append(i.baseUnit).append('\t').append(i.context).append('\t')
                    .append(i.syllable).append('\t').append(i.className).append('\t').append(i.occurrence).append('\t')
                    .append(String.format(Locale.US, "%.3f", i.prevCueMs)).append('\t')
                    .append(String.format(Locale.US, "%.3f", i.cueMs)).append('\t')
                    .append(String.format(Locale.US, "%.3f", i.nextCueMs)).append('\n');
        }
        return b.toString();
    }

    private String labelsTsv() {
        StringBuilder b = new StringBuilder();
        b.append("prompt_index\toccurrence\tanchor_ms_after_cue\tstatus\n");
        synchronized (labels) {
            for (Item item : items) {
                Label l = labels.get(item.key());
                if (l != null) {
                    b.append(item.promptIndex).append('\t').append(item.occurrence).append('\t')
                            .append(String.format(Locale.US, "%.3f", l.ms)).append('\t').append(l.status).append('\n');
                }
            }
        }
        return b.toString();
    }

    private static Map<String,String> parseForm(String body) throws UnsupportedEncodingException {
        Map<String,String> out = new HashMap<>();
        if (body == null || body.isBlank()) return out;
        for (String part : body.split("&")) {
            String[] kv = part.split("=", 2);
            String k = URLDecoder.decode(kv[0], StandardCharsets.UTF_8);
            String v = kv.length > 1 ? URLDecoder.decode(kv[1], StandardCharsets.UTF_8) : "";
            out.put(k, v);
        }
        return out;
    }

    private static Map<String,String> query(URI uri) throws UnsupportedEncodingException {
        return parseForm(uri.getRawQuery());
    }

    private static void send(HttpExchange ex, int status, String contentType, byte[] body) throws IOException {
        ex.getResponseHeaders().set("Content-Type", contentType);
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        ex.sendResponseHeaders(status, body.length);
        try (OutputStream out = ex.getResponseBody()) { out.write(body); }
    }

    private static void text(HttpExchange ex, int status, String contentType, String body) throws IOException {
        send(ex, status, contentType + "; charset=utf-8", body.getBytes(StandardCharsets.UTF_8));
    }

    private void handleLabel(HttpExchange ex) throws IOException {
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            text(ex, 405, "text/plain", "method not allowed"); return;
        }
        String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        Map<String,String> form = parseForm(body);
        try {
            int prompt = Integer.parseInt(form.get("prompt"));
            int occurrence = Integer.parseInt(form.get("occurrence"));
            String status = form.getOrDefault("status", "ok");
            String k = prompt + ":" + occurrence;
            if ("unset".equals(status)) {
                labels.remove(k);
            } else {
                double ms = Double.parseDouble(form.get("ms"));
                labels.put(k, new Label(ms, "uncertain".equals(status) ? "uncertain" : "ok"));
            }
            saveLabels();
            text(ex, 200, "text/plain", "ok");
        } catch (Exception err) {
            text(ex, 400, "text/plain", err.toString());
        }
    }

    private void handleAudio(HttpExchange ex) throws IOException {
        try {
            int prompt = Integer.parseInt(query(ex.getRequestURI()).getOrDefault("prompt", "-1"));
            Path wav = wavByPrompt.get(prompt);
            if (wav == null || !Files.isRegularFile(wav)) { text(ex,404,"text/plain","not found"); return; }
            send(ex, 200, "audio/wav", Files.readAllBytes(wav));
        } catch (Exception err) {
            text(ex, 400, "text/plain", err.toString());
        }
    }

    int run() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.setExecutor(Executors.newCachedThreadPool());
        server.createContext("/", ex -> text(ex, 200, "text/html", HTML));
        server.createContext("/manifest.tsv", ex -> text(ex, 200, "text/tab-separated-values", manifestTsv()));
        server.createContext("/labels.tsv", ex -> text(ex, 200, "text/tab-separated-values", labelsTsv()));
        server.createContext("/audio", this::handleAudio);
        server.createContext("/label", this::handleLabel);
        server.start();
        int port = server.getAddress().getPort();
        URI uri = URI.create("http://127.0.0.1:" + port + "/");
        System.out.println("PhonoWeave manual labels: " + uri);
        System.out.println("Ctrl-C to stop.");
        if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
            Desktop.getDesktop().browse(uri);
        }
        Thread.currentThread().join();
        return 0;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("usage: java --add-modules jdk.httpserver ManualAnchorWeb.java MANIFEST.tsv LABELS.tsv");
            System.exit(2);
        }
        new ManualAnchorWeb(Paths.get(args[0]).toAbsolutePath(), Paths.get(args[1]).toAbsolutePath()).run();
    }

    static final String HTML = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonoWeave</title>
<style>
:root{--bg:#eceee9;--panel:#fff;--line:#999d96;--ink:#191a18;--muted:#666962;--blue:#245b97;--purple:#73549a;--red:#b02c2c;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}.app{min-height:100vh;display:grid;grid-template-rows:36px 1fr 26px}.bar,.status{background:#dde0da;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 11px;gap:10px}.status{border-top:1px solid var(--line);border-bottom:0;font:11px var(--mono)}.brand{font-weight:800}.spacer{flex:1}.main{width:min(1120px,calc(100vw - 36px));margin:auto;padding:20px 0}.step{font:12px var(--mono);color:var(--muted)}.title{font-size:29px;font-weight:800;margin:5px 0 2px}.meta{font:13px var(--mono);color:#444;margin-bottom:13px}.panel{background:var(--panel);border:1px solid var(--line);padding:12px}.wave{display:block;width:100%;height:390px;border:1px solid #aaa;background:#fbfbfa;cursor:crosshair}.controls{display:flex;gap:7px;align-items:center;margin-top:10px}.controls button{border:1px solid #777b74;background:#fff;padding:7px 12px;font:inherit;cursor:pointer}.controls button.primary{background:#234f86;color:#fff;border-color:#234f86}.readout{font:12px var(--mono);margin-left:10px}.legend{font:11px var(--mono);color:var(--muted);margin-top:8px}.status .spacer{flex:1}
</style></head><body><div class="app"><div class="bar"><span class="brand">PHONOWEAVE</span><span>MANUAL LABELS</span></div><main class="main"><div class="step" id="step"></div><div class="title" id="title"></div><div class="meta" id="meta"></div><section class="panel"><canvas id="wave" class="wave"></canvas><div class="controls"><button id="prev">←</button><button id="play" class="primary">Play · Space</button><button id="uncertain">Uncertain · U</button><button id="clear">Clear · Del</button><button id="next">→</button><span class="readout" id="readout"></span></div><div class="legend">gray = prev beep · blue = cue · purple = next beep · red = mark</div></section></main><div class="status"><span id="status"></span><span class="spacer"></span><span id="count"></span></div></div>
<script>
const $=id=>document.getElementById(id);let items=[],labels=new Map(),index=0,ctx=null,buffers=new Map();
function parseTsv(s){const rows=s.trim().split(/\r?\n/).map(x=>x.split('\t'));const h=rows.shift();return rows.filter(r=>r.length>=h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])))}
function key(x){return `${x.prompt_index}:${x.occurrence}`}function cur(){return items[index]}
async function load(){items=parseTsv(await (await fetch('/manifest.tsv')).text()).map(x=>({...x,prompt_index:+x.prompt_index,occurrence:+x.occurrence,prev_cue_ms:+x.prev_cue_ms,cue_ms:+x.cue_ms,next_cue_ms:+x.next_cue_ms}));for(const x of parseTsv(await (await fetch('/labels.tsv')).text()))labels.set(`${x.prompt_index}:${x.occurrence}`,{ms:+x.anchor_ms_after_cue,status:x.status});await loadAudio(cur());render()}
async function loadAudio(x){if(buffers.has(x.prompt_index))return buffers.get(x.prompt_index);const arr=await (await fetch(`/audio?prompt=${x.prompt_index}`)).arrayBuffer();ctx??=new(window.AudioContext||window.webkitAudioContext)();const b=await ctx.decodeAudioData(arr.slice(0));buffers.set(x.prompt_index,b);return b}
function render(){const x=cur(),l=labels.get(key(x));$('step').textContent=`${index+1} / ${items.length}`;$('title').textContent=`${x.syllable}   ${x.occurrence}/2`;$('meta').textContent=`${x.base_unit} · ${x.context} · ${x.class_name}`;$('readout').textContent=l?`${l.ms>=0?'+':''}${l.ms.toFixed(1)} ms${l.status==='uncertain'?'  ?':''}`:'';$('uncertain').textContent=l?.status==='uncertain'?'Certain · U':'Uncertain · U';$('prev').disabled=index===0;$('next').disabled=index===items.length-1;$('count').textContent=`${labels.size}/${items.length}`;draw()}
function draw(){const c=$('wave'),g=c.getContext('2d'),d=devicePixelRatio||1,w=Math.max(1,Math.floor(c.clientWidth*d)),h=Math.max(1,Math.floor(c.clientHeight*d));if(c.width!==w||c.height!==h){c.width=w;c.height=h}g.clearRect(0,0,w,h);const x=cur(),b=buffers.get(x.prompt_index);if(!b)return;const a=Math.max(0,Math.floor(x.prev_cue_ms*b.sampleRate/1000)),z=Math.min(b.length,Math.ceil(x.next_cue_ms*b.sampleRate/1000)),span=Math.max(1,z-a),mid=h/2,step=Math.max(1,Math.floor(span/w));g.strokeStyle='#222';g.beginPath();for(let px=0;px<w;px++){let peak=0;const s=a+px*step,e=Math.min(z,s+step);for(let i=s;i<e;i++)peak=Math.max(peak,Math.abs(b.getChannelData(0)[i]));g.moveTo(px,mid-peak*h*.44);g.lineTo(px,mid+peak*h*.44)}g.stroke();line(x.prev_cue_ms,'#888','prev');line(x.cue_ms,'#245b97','cue');line(x.next_cue_ms,'#73549a','next');const l=labels.get(key(x));if(l)line(x.cue_ms+l.ms,'#b02c2c','',3);function line(ms,color,name,width=2){const px=(ms-x.prev_cue_ms)/(x.next_cue_ms-x.prev_cue_ms)*w;g.strokeStyle=color;g.lineWidth=width;g.beginPath();g.moveTo(px,0);g.lineTo(px,h);g.stroke();if(name){g.fillStyle=color;g.font=`${11*d}px ui-monospace`;g.fillText(name,Math.min(w-45*d,px+4*d),15*d)}g.lineWidth=1}}
async function save(x,ms,status){const k=key(x);if(status==='unset')labels.delete(k);else labels.set(k,{ms,status});render();const body=new URLSearchParams({prompt:String(x.prompt_index),occurrence:String(x.occurrence),ms:ms==null?'':String(ms),status});try{const r=await fetch('/label',{method:'POST',body});if(!r.ok)throw Error(await r.text());$('status').textContent='saved'}catch(e){$('status').textContent='SAVE ERROR: '+e}}
async function mark(ev){const c=$('wave'),r=c.getBoundingClientRect(),frac=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)),x=cur(),abs=x.prev_cue_ms+frac*(x.next_cue_ms-x.prev_cue_ms),ms=abs-x.cue_ms;save(x,ms,'ok');if(index<items.length-1){index++;await loadAudio(cur());render()}}
async function move(d){const n=Math.max(0,Math.min(items.length-1,index+d));if(n!==index){index=n;await loadAudio(cur());render()}}
async function play(){const x=cur(),b=await loadAudio(x);ctx??=new(window.AudioContext||window.webkitAudioContext)();await ctx.resume();const s=Math.max(0,(x.prev_cue_ms-60)/1000),e=Math.min(b.duration,(x.next_cue_ms+60)/1000),src=ctx.createBufferSource();src.buffer=b;src.connect(ctx.destination);src.start(0,s,e-s)}
function toggle(){const x=cur(),l=labels.get(key(x));if(l)save(x,l.ms,l.status==='uncertain'?'ok':'uncertain')}function clear(){const x=cur();if(labels.has(key(x)))save(x,null,'unset')}
$('wave').addEventListener('mousedown',mark);$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('play').onclick=play;$('uncertain').onclick=toggle;$('clear').onclick=clear;addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();play()}else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key.toLowerCase()==='u')toggle();else if(e.key==='Delete'||e.key==='Backspace')clear()});addEventListener('resize',render);load();
</script></body></html>
""";
}
