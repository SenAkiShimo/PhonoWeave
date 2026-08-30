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
            text(ex, 405, "text/plain", "method not allowed");
            return;
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
            if (wav == null || !Files.isRegularFile(wav)) {
                text(ex, 404, "text/plain", "not found");
                return;
            }
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
</style></head><body><div class="app"><div class="bar"><span class="brand">PHONOWEAVE</span><span>MANUAL LABELS</span></div><main class="main"><div class="step" id="step"></div><div class="title" id="title"></div><div class="meta" id="meta"></div><section class="panel"><canvas id="wave" class="wave"></canvas><div class="controls"><button id="prev">←</button><button id="play" class="primary">Play · Space</button><button id="uncertain">Uncertain · U</button><button id="clear">Clear · Del</button><button id="next">→</button><span class="readout" id="readout"></span></div><div class="legend">gray = prev beep · blue = cue · purple = next beep · red = mark</div></section></main><div class="status"><span id="status">loading</span><span class="spacer"></span><span id="count"></span></div></div>
<script>
const $=id=>document.getElementById(id);
let items=[];
let labels=new Map();
let index=0;
let audioCtx=null;
let buffers=new Map();

function splitLines(text){
  const cr=String.fromCharCode(13);
  const lf=String.fromCharCode(10);
  return text.replaceAll(cr,'').split(lf).filter(line=>line.length>0);
}

function parseTsv(text){
  const tab=String.fromCharCode(9);
  const lines=splitLines(text);
  if(lines.length===0)return [];
  const rows=lines.map(line=>line.split(tab));
  const header=rows.shift();
  return rows.filter(row=>row.length>=header.length).map(row=>Object.fromEntries(header.map((key,i)=>[key,row[i]])));
}

function itemKey(item){return item.prompt_index+':'+item.occurrence;}
function current(){return items[index];}

async function fetchText(url){
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw new Error(url+' HTTP '+response.status+' '+await response.text());
  return response.text();
}

async function loadAudio(item){
  if(!item)throw new Error('manifest is empty');
  if(buffers.has(item.prompt_index))return buffers.get(item.prompt_index);
  const response=await fetch('/audio?prompt='+encodeURIComponent(item.prompt_index),{cache:'no-store'});
  if(!response.ok)throw new Error('audio HTTP '+response.status+' '+await response.text());
  const arrayBuffer=await response.arrayBuffer();
  if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  const buffer=await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  buffers.set(item.prompt_index,buffer);
  return buffer;
}

async function initialize(){
  try{
    const manifestText=await fetchText('/manifest.tsv');
    items=parseTsv(manifestText).map(row=>({
      ...row,
      prompt_index:Number(row.prompt_index),
      occurrence:Number(row.occurrence),
      prev_cue_ms:Number(row.prev_cue_ms),
      cue_ms:Number(row.cue_ms),
      next_cue_ms:Number(row.next_cue_ms)
    }));
    const labelText=await fetchText('/labels.tsv');
    for(const row of parseTsv(labelText)){
      labels.set(row.prompt_index+':'+row.occurrence,{ms:Number(row.anchor_ms_after_cue),status:row.status});
    }
    if(items.length===0)throw new Error('manifest contains 0 items');
    await loadAudio(current());
    render();
    $('status').textContent='ready';
    prefetchNext();
  }catch(error){
    console.error(error);
    $('status').textContent='LOAD ERROR: '+error.message;
  }
}

function render(){
  const item=current();
  if(!item)return;
  const label=labels.get(itemKey(item));
  $('step').textContent=(index+1)+' / '+items.length;
  $('title').textContent=item.syllable+'   '+item.occurrence+'/2';
  $('meta').textContent=item.base_unit+' · '+item.context+' · '+item.class_name;
  $('readout').textContent=label?((label.ms>=0?'+':'')+label.ms.toFixed(1)+' ms'+(label.status==='uncertain'?'  ?':'')):'';
  $('uncertain').textContent=label&&label.status==='uncertain'?'Certain · U':'Uncertain · U';
  $('prev').disabled=index===0;
  $('next').disabled=index===items.length-1;
  $('count').textContent=labels.size+'/'+items.length;
  draw();
}

function draw(){
  const canvas=$('wave');
  const g=canvas.getContext('2d');
  const ratio=window.devicePixelRatio||1;
  const width=Math.max(1,Math.floor(canvas.clientWidth*ratio));
  const height=Math.max(1,Math.floor(canvas.clientHeight*ratio));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  g.clearRect(0,0,width,height);

  const item=current();
  const buffer=item?buffers.get(item.prompt_index):null;
  if(!item||!buffer)return;
  const samples=buffer.getChannelData(0);
  const start=Math.max(0,Math.floor(item.prev_cue_ms*buffer.sampleRate/1000));
  const end=Math.min(buffer.length,Math.ceil(item.next_cue_ms*buffer.sampleRate/1000));
  const span=Math.max(1,end-start);
  const mid=height/2;
  const samplesPerPixel=Math.max(1,Math.ceil(span/width));

  g.strokeStyle='#222';
  g.beginPath();
  for(let px=0;px<width;px++){
    const a=start+px*samplesPerPixel;
    if(a>=end)break;
    const z=Math.min(end,a+samplesPerPixel);
    let peak=0;
    for(let i=a;i<z;i++)peak=Math.max(peak,Math.abs(samples[i]));
    g.moveTo(px,mid-peak*height*0.44);
    g.lineTo(px,mid+peak*height*0.44);
  }
  g.stroke();

  drawLine(item.prev_cue_ms,'#888','prev',2);
  drawLine(item.cue_ms,'#245b97','cue',2);
  drawLine(item.next_cue_ms,'#73549a','next',2);
  const label=labels.get(itemKey(item));
  if(label)drawLine(item.cue_ms+label.ms,'#b02c2c','',3);

  function drawLine(ms,color,name,lineWidth){
    const px=(ms-item.prev_cue_ms)/(item.next_cue_ms-item.prev_cue_ms)*width;
    g.strokeStyle=color;
    g.lineWidth=lineWidth;
    g.beginPath();
    g.moveTo(px,0);
    g.lineTo(px,height);
    g.stroke();
    if(name){
      g.fillStyle=color;
      g.font=(11*ratio)+'px ui-monospace';
      g.fillText(name,Math.min(width-45*ratio,px+4*ratio),15*ratio);
    }
    g.lineWidth=1;
  }
}

function postLabel(item,ms,status){
  const body=new URLSearchParams({
    prompt:String(item.prompt_index),
    occurrence:String(item.occurrence),
    ms:ms==null?'':String(ms),
    status:status
  });
  fetch('/label',{method:'POST',body}).then(async response=>{
    if(!response.ok)throw new Error(await response.text());
    $('status').textContent='saved';
  }).catch(error=>{
    console.error(error);
    $('status').textContent='SAVE ERROR: '+error.message;
  });
}

async function mark(event){
  const canvas=$('wave');
  const rect=canvas.getBoundingClientRect();
  const fraction=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width));
  const item=current();
  const absoluteMs=item.prev_cue_ms+fraction*(item.next_cue_ms-item.prev_cue_ms);
  const relativeMs=absoluteMs-item.cue_ms;
  labels.set(itemKey(item),{ms:relativeMs,status:'ok'});
  postLabel(item,relativeMs,'ok');
  if(index<items.length-1){
    index++;
    try{
      await loadAudio(current());
      render();
      prefetchNext();
    }catch(error){
      $('status').textContent='AUDIO ERROR: '+error.message;
    }
  }else{
    render();
  }
}

async function move(delta){
  const nextIndex=Math.max(0,Math.min(items.length-1,index+delta));
  if(nextIndex===index)return;
  index=nextIndex;
  try{
    await loadAudio(current());
    render();
    prefetchNext();
  }catch(error){
    $('status').textContent='AUDIO ERROR: '+error.message;
  }
}

function prefetchNext(){
  const nextItem=items[index+1];
  if(nextItem)loadAudio(nextItem).catch(()=>{});
}

async function play(){
  try{
    const item=current();
    const buffer=await loadAudio(item);
    if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();
    await audioCtx.resume();
    const start=Math.max(0,(item.prev_cue_ms-60)/1000);
    const end=Math.min(buffer.duration,(item.next_cue_ms+60)/1000);
    const source=audioCtx.createBufferSource();
    source.buffer=buffer;
    source.connect(audioCtx.destination);
    source.start(0,start,Math.max(0.01,end-start));
  }catch(error){
    $('status').textContent='PLAY ERROR: '+error.message;
  }
}

function toggleUncertain(){
  const item=current();
  const label=labels.get(itemKey(item));
  if(!label)return;
  label.status=label.status==='uncertain'?'ok':'uncertain';
  render();
  postLabel(item,label.ms,label.status);
}

function clearMark(){
  const item=current();
  if(!labels.has(itemKey(item)))return;
  labels.delete(itemKey(item));
  render();
  postLabel(item,null,'unset');
}

$('wave').addEventListener('pointerdown',mark);
$('prev').addEventListener('click',()=>move(-1));
$('next').addEventListener('click',()=>move(1));
$('play').addEventListener('click',play);
$('uncertain').addEventListener('click',toggleUncertain);
$('clear').addEventListener('click',clearMark);
window.addEventListener('keydown',event=>{
  if(event.code==='Space'){event.preventDefault();play();}
  else if(event.key==='ArrowLeft')move(-1);
  else if(event.key==='ArrowRight')move(1);
  else if(event.key.toLowerCase()==='u')toggleUncertain();
  else if(event.key==='Delete'||event.key==='Backspace')clearMark();
});
window.addEventListener('resize',render);
window.addEventListener('error',event=>{$('status').textContent='JS ERROR: '+event.message;});
initialize();
</script></body></html>
""";
}
