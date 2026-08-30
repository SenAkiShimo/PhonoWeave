import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.awt.Desktop;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.Executors;

public class ManualAnchorWebV2 {
    static final class Item {
        int promptIndex; String baseUnit; String context; String syllable; String className; Path wav;
        int occurrence; double prevCueMs; double cueMs; double nextCueMs;
        String key(){ return promptIndex+":"+occurrence; }
    }
    static final class Label { double ms; String status; Label(double ms,String status){this.ms=ms;this.status=status;} }

    private final List<Item> items;
    private final Path labelsPath;
    private final String pythonExe;
    private final Path repoRoot;
    private final Map<String,Label> labels=Collections.synchronizedMap(new LinkedHashMap<>());
    private final Map<Integer,Path> wavByPrompt=new HashMap<>();
    private volatile Process guiProcess;
    private volatile String guiUrl;

    ManualAnchorWebV2(Path manifest,Path labelsPath,String pythonExe,Path repoRoot)throws IOException{
        this.items=readManifest(manifest);this.labelsPath=labelsPath;this.pythonExe=pythonExe;this.repoRoot=repoRoot;
        for(Item i:items)wavByPrompt.put(i.promptIndex,i.wav);loadLabels();
    }
    static List<Item> readManifest(Path path)throws IOException{
        List<Item> out=new ArrayList<>();
        for(String line:Files.readAllLines(path,StandardCharsets.UTF_8)){
            if(line.isBlank()||line.startsWith("#"))continue;String[] p=line.split("\\t",-1);if(p.length<10)throw new IOException("Bad manifest row");
            Item i=new Item();i.promptIndex=Integer.parseInt(p[0]);i.baseUnit=p[1];i.context=p[2];i.syllable=p[3];i.className=p[4];i.wav=Paths.get(p[5]);i.occurrence=Integer.parseInt(p[6]);i.prevCueMs=Double.parseDouble(p[7]);i.cueMs=Double.parseDouble(p[8]);i.nextCueMs=Double.parseDouble(p[9]);out.add(i);
        }return out;
    }
    void loadLabels()throws IOException{
        if(!Files.isRegularFile(labelsPath))return;
        for(String line:Files.readAllLines(labelsPath,StandardCharsets.UTF_8)){
            if(line.isBlank()||line.startsWith("#"))continue;String[] p=line.split("\\t",-1);if(p.length<4)continue;
            try{labels.put(p[0]+":"+p[1],new Label(Double.parseDouble(p[2]),p[3]));}catch(NumberFormatException ignored){}
        }
    }
    synchronized void saveLabels()throws IOException{
        Files.createDirectories(labelsPath.getParent());List<String> out=new ArrayList<>();out.add("# prompt_index\toccurrence\tanchor_ms_after_cue\tstatus");
        for(Item i:items){Label l=labels.get(i.key());if(l!=null)out.add(String.format(Locale.US,"%d\t%d\t%.3f\t%s",i.promptIndex,i.occurrence,l.ms,l.status));}
        Path tmp=labelsPath.resolveSibling(labelsPath.getFileName()+".tmp");Files.write(tmp,out,StandardCharsets.UTF_8);Files.move(tmp,labelsPath,StandardCopyOption.REPLACE_EXISTING);
    }
    String manifestTsv(){StringBuilder b=new StringBuilder("prompt_index\tbase_unit\tcontext\tsyllable\tclass_name\toccurrence\tprev_cue_ms\tcue_ms\tnext_cue_ms\n");for(Item i:items)b.append(i.promptIndex).append('\t').append(i.baseUnit).append('\t').append(i.context).append('\t').append(i.syllable).append('\t').append(i.className).append('\t').append(i.occurrence).append('\t').append(String.format(Locale.US,"%.3f",i.prevCueMs)).append('\t').append(String.format(Locale.US,"%.3f",i.cueMs)).append('\t').append(String.format(Locale.US,"%.3f",i.nextCueMs)).append('\n');return b.toString();}
    String labelsTsv(){StringBuilder b=new StringBuilder("prompt_index\toccurrence\tanchor_ms_after_cue\tstatus\n");synchronized(labels){for(Item i:items){Label l=labels.get(i.key());if(l!=null)b.append(i.promptIndex).append('\t').append(i.occurrence).append('\t').append(String.format(Locale.US,"%.3f",l.ms)).append('\t').append(l.status).append('\n');}}return b.toString();}
    static Map<String,String> parseForm(String body)throws UnsupportedEncodingException{Map<String,String> out=new HashMap<>();if(body==null||body.isBlank())return out;for(String part:body.split("&")){String[] kv=part.split("=",2);out.put(URLDecoder.decode(kv[0],StandardCharsets.UTF_8),kv.length>1?URLDecoder.decode(kv[1],StandardCharsets.UTF_8):"");}return out;}
    static void send(HttpExchange ex,int status,String type,byte[] body)throws IOException{ex.getResponseHeaders().set("Content-Type",type);ex.getResponseHeaders().set("Cache-Control","no-store");ex.sendResponseHeaders(status,body.length);try(OutputStream o=ex.getResponseBody()){o.write(body);}}
    static void text(HttpExchange ex,int status,String type,String body)throws IOException{send(ex,status,type+"; charset=utf-8",body.getBytes(StandardCharsets.UTF_8));}
    void handleLabel(HttpExchange ex)throws IOException{String body=new String(ex.getRequestBody().readAllBytes(),StandardCharsets.UTF_8);Map<String,String> f=parseForm(body);try{String k=f.get("prompt")+":"+f.get("occurrence");String st=f.getOrDefault("status","ok");if("unset".equals(st))labels.remove(k);else labels.put(k,new Label(Double.parseDouble(f.get("ms")),"uncertain".equals(st)?"uncertain":"ok"));saveLabels();text(ex,200,"text/plain","ok");}catch(Exception e){text(ex,400,"text/plain",e.toString());}}
    void handleAudio(HttpExchange ex)throws IOException{try{Map<String,String> q=parseForm(ex.getRequestURI().getRawQuery());Path wav=wavByPrompt.get(Integer.parseInt(q.getOrDefault("prompt","-1")));if(wav==null||!Files.isRegularFile(wav)){text(ex,404,"text/plain","not found");return;}send(ex,200,"audio/wav",Files.readAllBytes(wav));}catch(Exception e){text(ex,400,"text/plain",e.toString());}}
    synchronized String ensureGui()throws IOException{
        if(guiUrl!=null&&guiProcess!=null&&guiProcess.isAlive())return guiUrl;
        ProcessBuilder pb=new ProcessBuilder(pythonExe,"-m","phonoweave.gui","--no-browser");pb.directory(repoRoot.toFile());pb.redirectErrorStream(true);guiProcess=pb.start();
        BufferedReader r=new BufferedReader(new InputStreamReader(guiProcess.getInputStream(),StandardCharsets.UTF_8));long end=System.currentTimeMillis()+5000;
        while(System.currentTimeMillis()<end){String line=r.readLine();if(line==null)break;if(line.startsWith("PhonoWeave GUI remake: ")){guiUrl=line.substring("PhonoWeave GUI remake: ".length()).trim();break;}}
        if(guiUrl==null)throw new IOException("Could not start PhonoWeave GUI");return guiUrl;
    }
    int run()throws Exception{
        HttpServer s=HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(),0),0);s.setExecutor(Executors.newCachedThreadPool());
        s.createContext("/",ex->text(ex,200,"text/html",HTML));s.createContext("/manifest.tsv",ex->text(ex,200,"text/tab-separated-values",manifestTsv()));s.createContext("/labels.tsv",ex->text(ex,200,"text/tab-separated-values",labelsTsv()));s.createContext("/audio",this::handleAudio);s.createContext("/label",this::handleLabel);
        s.createContext("/return",ex->{try{text(ex,200,"text/plain",ensureGui());}catch(Exception e){text(ex,500,"text/plain",e.toString());}});
        s.start();URI uri=URI.create("http://127.0.0.1:"+s.getAddress().getPort()+"/");System.out.println("PhonoWeave manual labels: "+uri);System.out.println("Ctrl-C to stop.");if(Desktop.isDesktopSupported()&&Desktop.getDesktop().isSupported(Desktop.Action.BROWSE))Desktop.getDesktop().browse(uri);Thread.currentThread().join();return 0;
    }
    public static void main(String[] args)throws Exception{if(args.length!=4){System.err.println("usage: ManualAnchorWebV2 MANIFEST LABELS PYTHON REPO");System.exit(2);}new ManualAnchorWebV2(Paths.get(args[0]),Paths.get(args[1]),args[2],Paths.get(args[3])).run();}

    static final String HTML="""
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhonoWeave</title>
<style>
:root{--bg:#eceee9;--panel:#fff;--line:#999d96;--ink:#191a18;--muted:#666962;--blue:#245b97;--purple:#73549a;--red:#b02c2c;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}.app{min-height:100vh;display:grid;grid-template-rows:38px 1fr 26px}.bar,.status{background:#dde0da;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 11px;gap:10px}.status{border-top:1px solid var(--line);border-bottom:0;font:11px var(--mono)}.brand{font-weight:800}.spacer{flex:1}.main{width:min(1180px,calc(100vw - 36px));margin:auto;padding:18px 0}.step{font:12px var(--mono);color:var(--muted)}.title{font-size:28px;font-weight:800;margin:4px 0 2px}.meta{font:13px var(--mono);color:#444;margin-bottom:10px}.panel{background:var(--panel);border:1px solid var(--line);padding:10px}.wave,.spec{display:block;width:100%;border:1px solid #aaa;background:#fbfbfa;cursor:crosshair}.wave{height:260px}.spec{height:250px;margin-top:6px;background:#111}.controls{display:flex;gap:7px;align-items:center;margin-top:9px}.controls button,.bar button{border:1px solid #777b74;background:#fff;padding:6px 11px;font:inherit;cursor:pointer}.controls button.primary{background:#234f86;color:#fff;border-color:#234f86}.readout{font:12px var(--mono);margin-left:10px}.legend{font:11px var(--mono);color:var(--muted);margin-top:7px}.done{position:fixed;inset:0;background:rgba(0,0,0,.36);display:none;align-items:center;justify-content:center}.done.show{display:flex}.donebox{background:#fff;border:1px solid #777;padding:24px 28px;min-width:340px;text-align:center}.donebox h2{margin:0 0 8px}.donebox button{margin-top:14px;padding:7px 13px}.bar .spacer,.status .spacer{flex:1}
</style></head><body><div class="app"><div class="bar"><span class="brand">PHONOWEAVE</span><span>MANUAL LABELS</span><span class="spacer"></span><button id="back">← PhonoWeave</button></div><main class="main"><div class="step" id="step"></div><div class="title" id="title"></div><div class="meta" id="meta"></div><section class="panel"><canvas id="wave" class="wave"></canvas><canvas id="spec" class="spec"></canvas><div class="controls"><button id="prev">←</button><button id="play" class="primary">Play · Space</button><button id="uncertain">Uncertain · U</button><button id="clear">Clear · Del</button><button id="next">→</button><span class="readout" id="readout"></span></div><div class="legend">gray = prev beep · blue = cue · purple = next beep · red = mark</div></section></main><div class="status"><span id="status">loading</span><span class="spacer"></span><span id="count"></span></div></div><div id="done" class="done"><div class="donebox"><h2>32 / 32 complete</h2><div>Manual labels saved.</div><button id="doneBack">Return to PhonoWeave</button><button id="doneClose">Stay here</button></div></div>
<script>
const $=id=>document.getElementById(id);let items=[],labels=new Map(),index=0,audioCtx=null,buffers=new Map(),completionShown=false;
function splitLines(t){const cr=String.fromCharCode(13),lf=String.fromCharCode(10);return t.replaceAll(cr,'').split(lf).filter(x=>x.length>0)}function parseTsv(t){const tab=String.fromCharCode(9),rows=splitLines(t).map(x=>x.split(tab)),h=rows.shift();if(!h)return[];return rows.filter(r=>r.length>=h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k,r[i]])))}function key(x){return x.prompt_index+':'+x.occurrence}function cur(){return items[index]}
async function text(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(await r.text());return r.text()}async function loadAudio(x){if(buffers.has(x.prompt_index))return buffers.get(x.prompt_index);const r=await fetch('/audio?prompt='+x.prompt_index);if(!r.ok)throw Error(await r.text());audioCtx??=new(window.AudioContext||window.webkitAudioContext)();const b=await audioCtx.decodeAudioData((await r.arrayBuffer()).slice(0));buffers.set(x.prompt_index,b);return b}
async function init(){try{items=parseTsv(await text('/manifest.tsv')).map(x=>({...x,prompt_index:+x.prompt_index,occurrence:+x.occurrence,prev_cue_ms:+x.prev_cue_ms,cue_ms:+x.cue_ms,next_cue_ms:+x.next_cue_ms}));for(const x of parseTsv(await text('/labels.tsv')))labels.set(x.prompt_index+':'+x.occurrence,{ms:+x.anchor_ms_after_cue,status:x.status});await loadAudio(cur());render();$('status').textContent='ready';prefetch();checkDone(true)}catch(e){$('status').textContent='LOAD ERROR: '+e.message}}
function render(){const x=cur(),l=labels.get(key(x));$('step').textContent=(index+1)+' / '+items.length;$('title').textContent=x.syllable+'   '+x.occurrence+'/2';$('meta').textContent=x.base_unit+' · '+x.context+' · '+x.class_name;$('readout').textContent=l?((l.ms>=0?'+':'')+l.ms.toFixed(1)+' ms'+(l.status==='uncertain'?'  ?':'')):'';$('uncertain').textContent=l&&l.status==='uncertain'?'Certain · U':'Uncertain · U';$('prev').disabled=index===0;$('next').disabled=index===items.length-1;$('count').textContent=labels.size+'/'+items.length;drawWave();drawSpec()}
function setupCanvas(id){const c=$(id),g=c.getContext('2d'),d=devicePixelRatio||1,w=Math.max(1,Math.floor(c.clientWidth*d)),h=Math.max(1,Math.floor(c.clientHeight*d));if(c.width!==w||c.height!==h){c.width=w;c.height=h}return{c,g,d,w,h}}
function spanData(){const x=cur(),b=buffers.get(x.prompt_index),s=b.getChannelData(0),a=Math.max(0,Math.floor(x.prev_cue_ms*b.sampleRate/1000)),z=Math.min(b.length,Math.ceil(x.next_cue_ms*b.sampleRate/1000));return{x,b,s,a,z}}
function lines(g,w,h,x,d){function ln(ms,col,name,lw=2){const px=(ms-x.prev_cue_ms)/(x.next_cue_ms-x.prev_cue_ms)*w;g.strokeStyle=col;g.lineWidth=lw;g.beginPath();g.moveTo(px,0);g.lineTo(px,h);g.stroke();if(name){g.fillStyle=col;g.font=(10*d)+'px ui-monospace';g.fillText(name,Math.min(w-40*d,px+3*d),13*d)}g.lineWidth=1}ln(x.prev_cue_ms,'#888','prev');ln(x.cue_ms,'#245b97','cue');ln(x.next_cue_ms,'#73549a','next');const l=labels.get(key(x));if(l)ln(x.cue_ms+l.ms,'#d23b3b','',3)}
function drawWave(){const {g,d,w,h}=setupCanvas('wave'),{x,s,a,z}=spanData(),span=Math.max(1,z-a),mid=h/2,spp=Math.max(1,Math.ceil(span/w));g.clearRect(0,0,w,h);g.strokeStyle='#222';g.beginPath();for(let px=0;px<w;px++){const aa=a+px*spp;if(aa>=z)break;const zz=Math.min(z,aa+spp);let p=0;for(let i=aa;i<zz;i++)p=Math.max(p,Math.abs(s[i]));g.moveTo(px,mid-p*h*.44);g.lineTo(px,mid+p*h*.44)}g.stroke();lines(g,w,h,x,d)}
function fft(re,im){for(let i=1,j=0;i<re.length;i++){let bit=re.length>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]]}}for(let len=2;len<=re.length;len<<=1){const ang=-2*Math.PI/len;for(let i=0;i<re.length;i+=len){for(let j=0;j<len/2;j++){const c=Math.cos(ang*j),s=Math.sin(ang*j),ur=re[i+j],ui=im[i+j],vr=re[i+j+len/2]*c-im[i+j+len/2]*s,vi=re[i+j+len/2]*s+im[i+j+len/2]*c;re[i+j]=ur+vr;im[i+j]=ui+vi;re[i+j+len/2]=ur-vr;im[i+j+len/2]=ui-vi}}}}
function drawSpec(){const {g,d,w,h}=setupCanvas('spec'),{x,b,s,a,z}=spanData(),N=256,hop=Math.max(32,Math.floor((z-a)/Math.max(1,Math.floor(w/d)))),frames=Math.max(1,Math.floor((z-a-N)/hop)+1),bins=N/2,img=g.createImageData(frames,bins),mags=[];let max=-1e9,min=1e9;for(let f=0;f<frames;f++){const re=new Array(N),im=new Array(N).fill(0),off=a+f*hop;for(let n=0;n<N;n++)re[n]=(s[off+n]||0)*(0.5-0.5*Math.cos(2*Math.PI*n/(N-1)));fft(re,im);const row=[];for(let k=0;k<bins;k++){const m=20*Math.log10(Math.hypot(re[k],im[k])+1e-6);row.push(m);max=Math.max(max,m);min=Math.min(min,m)}mags.push(row)}const floor=max-70;for(let f=0;f<frames;f++)for(let k=0;k<bins;k++){const v=Math.max(0,Math.min(1,(mags[f][k]-floor)/70)),y=bins-1-k,p=(y*frames+f)*4,q=Math.floor(v*255);img.data[p]=q;img.data[p+1]=q;img.data[p+2]=q;img.data[p+3]=255}const tmp=document.createElement('canvas');tmp.width=frames;tmp.height=bins;tmp.getContext('2d').putImageData(img,0,0);g.imageSmoothingEnabled=false;g.clearRect(0,0,w,h);g.drawImage(tmp,0,0,w,h);lines(g,w,h,x,d)}
function post(x,ms,status){const body=new URLSearchParams({prompt:String(x.prompt_index),occurrence:String(x.occurrence),ms:ms==null?'':String(ms),status});fetch('/label',{method:'POST',body}).then(r=>{if(!r.ok)throw Error('save');$('status').textContent='saved'}).catch(e=>$('status').textContent='SAVE ERROR')}
async function mark(ev){const c=ev.currentTarget,r=c.getBoundingClientRect(),frac=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)),x=cur(),abs=x.prev_cue_ms+frac*(x.next_cue_ms-x.prev_cue_ms),ms=abs-x.cue_ms;labels.set(key(x),{ms,status:'ok'});post(x,ms,'ok');if(index<items.length-1){index++;await loadAudio(cur());render();prefetch()}else render();checkDone(false)}async function move(d){const n=Math.max(0,Math.min(items.length-1,index+d));if(n===index)return;index=n;await loadAudio(cur());render();prefetch()}function prefetch(){const n=items[index+1];if(n)loadAudio(n).catch(()=>{})}async function play(){const x=cur(),b=await loadAudio(x);await audioCtx.resume();const s=Math.max(0,(x.prev_cue_ms-60)/1000),e=Math.min(b.duration,(x.next_cue_ms+60)/1000),src=audioCtx.createBufferSource();src.buffer=b;src.connect(audioCtx.destination);src.start(0,s,e-s)}function toggle(){const x=cur(),l=labels.get(key(x));if(!l)return;l.status=l.status==='uncertain'?'ok':'uncertain';render();post(x,l.ms,l.status)}function clear(){const x=cur();if(!labels.has(key(x)))return;labels.delete(key(x));render();post(x,null,'unset')}
function checkDone(onLoad){if(labels.size===items.length&&!completionShown){completionShown=true;$('done').classList.add('show');$('status').textContent='complete'}}async function goBack(){try{$('status').textContent='opening PhonoWeave';const r=await fetch('/return',{method:'POST'});if(!r.ok)throw Error(await r.text());location.href=(await r.text()).trim()}catch(e){$('status').textContent='RETURN ERROR: '+e.message}}
$('wave').addEventListener('pointerdown',mark);$('spec').addEventListener('pointerdown',mark);$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('play').onclick=play;$('uncertain').onclick=toggle;$('clear').onclick=clear;$('back').onclick=goBack;$('doneBack').onclick=goBack;$('doneClose').onclick=()=>$('done').classList.remove('show');addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();play()}else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key.toLowerCase()==='u')toggle();else if(e.key==='Delete'||e.key==='Backspace')clear()});addEventListener('resize',render);init();
</script></body></html>
""";
}
