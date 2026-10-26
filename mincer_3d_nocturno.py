# -*- coding: utf-8 -*-
"""
MINCER COLOMBIA — retornos a la educación con microdatos REALES de la GEIH-DANE + visualización nocturna 3D/AR.
ARCHIVO ÚNICO Y AUTOCONTENIDO (pegar completo en UNA celda de Google Colab y ejecutar; o  python mincer_3d_nocturno.py).

Qué hace
  1) Descarga los microdatos mensuales de la GEIH (marco 2018) del ANDA-DANE (microdatos.dane.gov.co) y la geometría oficial MGN-DANE.
  2) Construye ln(salario/hora), experiencia potencial (edad − esc − 6) y estima por MCO con errores robustos HC1/cluster:
         ln(w) = b0 + b1·esc + b2·exp + b3·exp² + b4·mujer (+ efectos fijos de departamento)
  3) Lanza una interfaz Gradio de "visión nocturna": mapa 2D, escena 3D/AR extruida sobre el shape departamental (dinámica mensual),
     superficie 3D y nube 3D de la ecuación, e interpretación automática (incluye Moran's I de los retornos).

Variables oficiales GEIH usadas (verificadas en el Diccionario de Datos GEIH 2024):
  Características generales: DIRECTORIO, SECUENCIA_P, ORDEN, MES, DPTO, AREA, CLASE, FEX_C18, P3271 (sexo), P6040 (edad), P3042 (nivel educativo)
  Ocupados: INGLABO (ingresos laborales), P6800 (horas semanales en ese trabajo)
  Catálogos ANDA verificados: 2022=771, 2023=782, 2024=819, 2026=900.

Supuestos y límites (cero invención):
  * En marco 2018 NO existen P6210, P6020 ni ESC. Los años de escolaridad se CONSTRUYEN desde P3042 (duración nominal por nivel).
  * P3271: el diccionario público solo da el rango 1–2; se asume 1 = hombre, 2 = mujer (el código avisa si el signo es inesperado).
  * MCO no es causal (habilidad no observada, selección). INGLABO es mensual total y P6800 solo el empleo principal.
  * Si el portal exige licencia/login, copie los .zip mensuales a data/geih/<año>/ y vuelva a ejecutar.
"""
import io, re, json, html, time, zipfile, unicodedata, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")


def _instalar():
    """Instala solo lo que falte (útil en Colab). Se llama desde main()."""
    import importlib.util, subprocess, sys
    req = {"geopandas": "geopandas", "requests": "requests", "statsmodels": "statsmodels", "gradio": "gradio", "plotly": "plotly"}
    faltan = [p for p, m in req.items() if importlib.util.find_spec(m) is None]
    if faltan:
        print("Instalando:", ", ".join(faltan)); subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltan])


# ============================================================ BLOQUE A — utilidades, geometría oficial MGN-DANE y escena 3D/AR
UA_DANE = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "es-CO,es;q=0.9,en;q=0.8", "Referer": "https://www.dane.gov.co/"}

MGN_URL = ("https://ags.esri.co/arcgis/rest/services/DatosAbiertos/MGN_Departamentos/MapServer/0/query"
           "?where=1%3D1&outFields=DPTO_CCDGO%2CDPTO_CNMBR&outSR=4326&maxAllowableOffset=0.005&geometryPrecision=4&f=geojson")

class _Resp:                                                     # mismo contrato que requests.Response (lo que usamos)
    def __init__(self, content, status=200): self.content, self.status_code = content, status; self.text = content.decode("utf-8", "ignore")

def _get(url, retries=2, timeout=90):
    """Prueba requests (UA de navegador) y, si falla, urllib (el cliente que funcionó en el notebook original)."""
    import urllib.request, urllib.error
    errs = []
    for i in range(retries):
        try:
            import requests
            r = requests.get(url, headers=UA_DANE, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 500: return r
            errs.append(f"requests HTTP {r.status_code}")
        except Exception as ex: errs.append(f"requests {type(ex).__name__}: {str(ex)[:80]}")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA_DANE), timeout=timeout) as u: data = u.read()
            if len(data) > 500: return _Resp(data)
            errs.append("urllib respuesta vacía")
        except urllib.error.HTTPError as ex: errs.append(f"urllib HTTP {ex.code}")
        except Exception as ex: errs.append(f"urllib {type(ex).__name__}: {str(ex)[:80]}")
        if any(c in e for e in errs for c in ("404", "403")) and i == 0 and "404" in " ".join(errs): break
        time.sleep(1.5 * (i + 1))
    raise IOError(f"{url} → " + " | ".join(errs))

def download(url, dest, force=False):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 500 and not force: return dest
    dest.write_bytes(_get(url).content); return dest

def fetch_mgn(outdir="data"):
    """Departamentos del Marco Geoestadístico Nacional (DANE; fuente básica IGAC): 32 departamentos + Bogotá D.C. = 33.
    Servido por el ArcGIS REST de DatosAbiertos (copyright DANE); descarga completa en https://geoportal.dane.gov.co (MGN)."""
    import geopandas as gpd
    p = download(MGN_URL, Path(outdir) / "mgn_departamentos.geojson")
    gj = json.loads(p.read_text(encoding="utf-8"))
    if "features" not in gj:
        p.unlink(missing_ok=True); raise IOError("El servicio MGN no devolvió 'features': " + str(gj)[:200])
    g = gpd.GeoDataFrame.from_features(gj["features"], crs=4326).rename(columns={"DPTO_CNMBR": "departamento", "DPTO_CCDGO": "codigo"})
    g["key"] = g["departamento"].map(norm_key)
    if len(g) < 32: raise IOError(f"MGN devolvió {len(g)} unidades (esperadas 33)")
    return g[["codigo", "departamento", "key", "geometry"]]

def norm_key(s):
    """Clave canónica: sin tildes/espacios/puntuación (las fuentes escriben 'BOGOTÁ, D.C.', 'BOGOTAD.C', 'LAGUAJIRA'...)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    k = re.sub(r"[^A-Z0-9]", "", s)
    if k.startswith("BOGOTA"): return "BOGOTADC"
    if k.startswith("SANANDRES") or k.startswith("ARCHIPIELAGODESANANDRES"): return "SANANDRESYPROVIDENCIA"
    return k

def row_std(A):
    s = A.sum(1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A, dtype=float), where=s > 0)

def moran(v, W, perm=999, seed=0):
    z = (v - v.mean()) / v.std(); I = z @ W @ z / (z @ z)
    rng = np.random.default_rng(seed); sim = np.array([(lambda p: p @ W @ p / (p @ p))(rng.permutation(z)) for _ in range(perm)])
    return I, (1 + (np.abs(sim) >= abs(I)).sum()) / (perm + 1)

DATA = "data"

def load_geometry():
    """Límites oficiales DANE-MGN (ver spvar_data.fetch_mgn). Devuelve geometría ligera + matrices auxiliares."""
    from spvar_data import fetch_mgn
    g = fetch_mgn(DATA); gp = g.to_crs(9377)               # MAGNA-SIRGAS Origen Nacional (metros)
    xy = np.c_[gp.geometry.centroid.x, gp.geometry.centroid.y]
    geoms = list(gp.geometry.buffer(1000)); n = len(g); adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): adj[i, j] = adj[j, i] = float(geoms[i].intersects(geoms[j]))
    gj = json.loads(g[["key", "codigo", "departamento", "geometry"]].to_json())
    return dict(names=list(g["departamento"]), keys=list(g["key"]), codigos=list(g["codigo"]), xy=xy, adj=adj, geojson=gj)

def scene_html(geojson, sets, height=680):
    """geojson: FeatureCollection (orden = filas de sets[*]['vals'][f]); sets: {clave:{label,frames,vals,unit}}"""
    feats = []
    for f in geojson["features"]:
        g = f["geometry"]; polys = [] if not g else ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"])
        feats.append(dict(n=f["properties"]["departamento"], polys=[[[[round(x, 3), round(y, 3)] for x, y in ring[::1]] for ring in p] for p in polys]))
    data = json.dumps(dict(features=feats, sets=sets), allow_nan=False, default=lambda o: None)
    page = SCENE_TPL.replace("__DATA__", data)
    return (f'<iframe style="width:100%;height:{height}px;border:1px solid #1c5;border-radius:6px;background:#000" '
            f'allow="xr-spatial-tracking; camera; gyroscope; accelerometer; fullscreen" srcdoc="{html.escape(page, quote=True)}"></iframe>')

SCENE_TPL = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden;font:12px 'Courier New',monospace;color:#6dff5a}
#c{position:absolute;inset:0}canvas{display:block}video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:grayscale(1) contrast(1.4) brightness(1.2) sepia(1) hue-rotate(60deg) saturate(3);display:none}
#nv{position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at center,transparent 45%,rgba(0,20,0,.85) 100%),repeating-linear-gradient(0deg,rgba(0,255,60,.05) 0 1px,transparent 1px 3px);animation:fl 5s infinite}
@keyframes fl{0%,100%{opacity:1}50%{opacity:.88}93%{opacity:.7}}
#ui{position:absolute;left:8px;top:8px;right:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;z-index:5}
select,button,input{background:#031a08;color:#6dff5a;border:1px solid #1c5;padding:4px 7px;font:inherit;border-radius:3px}button:hover{background:#0a3a14}
#tip{position:absolute;pointer-events:none;background:#031a08ee;border:1px solid #1c5;padding:5px 8px;display:none;z-index:6;white-space:pre}
#leg{position:absolute;left:10px;bottom:10px;z-index:5;text-shadow:0 0 6px #0f0}#bar{width:160px;height:9px;background:linear-gradient(90deg,#03140a,#0b7a2b,#5dff4a,#eaffd8);border:1px solid #1c5}
</style>
<video id=v playsinline muted></video><div id=c></div><div id=nv></div>
<div id=ui><select id=set></select><button id=pl>▶</button><input id=sl type=range min=0 max=0 value=0 style="width:150px"><span id=fl></span>
<button id=top>Cenital</button><button id=ar>AR</button><button id=cam>AR cámara</button></div>
<div id=tip></div><div id=leg><div id=lt></div><div id=bar></div><div style="display:flex;justify-content:space-between"><span id=lo></span><span id=hi></span></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D=__DATA__,$=i=>document.getElementById(i);
const R=new THREE.WebGLRenderer({antialias:true,alpha:true});R.setPixelRatio(devicePixelRatio);$('c').appendChild(R.domElement);
const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(45,1,.001,1e4);
S.add(new THREE.AmbientLight(0x3a6a3a,1.1));const dl=new THREE.DirectionalLight(0xaaffaa,1.1);dl.position.set(3,6,4);S.add(dl);
const holder=new THREE.Group(),root=new THREE.Group(),pivot=new THREE.Group();root.rotation.x=-Math.PI/2;holder.add(root);root.add(pivot);S.add(holder);
let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
D.features.forEach(f=>f.polys.forEach(p=>p[0].forEach(c=>{x0=Math.min(x0,c[0]);x1=Math.max(x1,c[0]);y0=Math.min(y0,c[1]);y1=Math.max(y1,c[1])})));
const EXT=Math.max(x1-x0,y1-y0),cx=(x0+x1)/2,cy=(y0+y1)/2;pivot.position.set(-cx,-cy,0);
const M=[];
D.features.forEach((f,i)=>{const g=new THREE.Group(),mat=new THREE.MeshStandardMaterial({color:0x0a3a14,flatShading:true,roughness:.55,metalness:.15});
 f.polys.forEach(p=>{const V=a=>a.map(c=>new THREE.Vector2(c[0],c[1])),sh=new THREE.Shape(V(p[0]));p.slice(1).forEach(h=>sh.holes.push(new THREE.Path(V(h))));
  const geo=new THREE.ExtrudeGeometry(sh,{depth:1,bevelEnabled:false}),m=new THREE.Mesh(geo,mat);m.userData.i=i;g.add(m);
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo,35),new THREE.LineBasicMaterial({color:0x7cff6b,transparent:true,opacity:.55})))});
 pivot.add(g);M.push({g,mat,h:.02,th:.02,col:new THREE.Color(0x1a1f1a)})});
const ZS=EXT*.22,ramp=[[0,'#03140a'],[.35,'#0b7a2b'],[.7,'#5dff4a'],[1,'#eaffd8']];
function rc(t){t=Math.min(1,Math.max(0,t));for(let i=1;i<ramp.length;i++)if(t<=ramp[i][0]){const a=ramp[i-1],b=ramp[i];return new THREE.Color(a[1]).lerp(new THREE.Color(b[1]),(t-a[0])/(b[0]-a[0]))}return new THREE.Color(ramp[3][1])}
let cur=null,fr=0,mn=0,mx=1;
Object.keys(D.sets).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=D.sets[k].label;$('set').appendChild(o)});
function useSet(k){cur=D.sets[k];const a=[].concat(...cur.vals).filter(v=>v!==null);mn=Math.min(...a);mx=Math.max(...a);if(mx==mn)mx=mn+1e-9;
 $('sl').max=cur.frames.length-1;$('sl').value=0;$('lt').textContent=cur.label+(cur.unit?' ['+cur.unit+']':'');$('lo').textContent=mn.toPrecision(3);$('hi').textContent=mx.toPrecision(3);frame(0)}
function frame(f){fr=f;$('sl').value=f;$('fl').textContent=cur.frames[f];cur.vals[f].forEach((v,i)=>{const m=M[i];if(v===null){m.th=.01;m.col=new THREE.Color(0x1a1f1a)}else{const t=(v-mn)/(mx-mn);m.th=.03+t;m.col=rc(t)}})}
$('set').onchange=e=>useSet(e.target.value);$('sl').oninput=e=>frame(+e.target.value);
let play=true,tm=setInterval(()=>{if(play&&cur)frame((fr+1)%cur.frames.length)},1100);$('pl').onclick=()=>{play=!play;$('pl').textContent=play?'⏸':'▶'};$('pl').textContent='⏸';
let az=.6,po=1.0,rad=2.1*EXT,drag=0,px=0,py=0,mode='orbit';
R.domElement.onpointerdown=e=>{drag=1;px=e.clientX;py=e.clientY};onpointerup=()=>drag=0;
onpointermove=e=>{if(drag&&mode=='orbit'){az-=(e.clientX-px)*.008;po=Math.min(1.55,Math.max(.05,po-(e.clientY-py)*.008));px=e.clientX;py=e.clientY}hover(e)};
R.domElement.onwheel=e=>{rad*=e.deltaY>0?1.08:.92;e.preventDefault()};
$('top').onclick=()=>{po=.05;az=0};
const RC=new THREE.Raycaster(),mv=new THREE.Vector2();
function hover(e){if(mode!='orbit')return;const r=R.domElement.getBoundingClientRect();mv.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);RC.setFromCamera(mv,C);
 const hit=RC.intersectObjects(pivot.children.flatMap(g=>g.children).filter(o=>o.isMesh))[0],t=$('tip');
 if(hit&&cur){const i=hit.object.userData.i,v=cur.vals[fr][i];t.style.display='block';t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px';t.textContent=D.features[i].n+'\n'+(v===null?'sin dato':v.toFixed(4))}else t.style.display='none'}
function size(){const w=innerWidth,h=innerHeight;R.setSize(w,h);C.aspect=w/h;C.updateProjectionMatrix()}addEventListener('resize',size);size();
function tick(){M.forEach(m=>{m.h+=(m.th-m.h)*.12;m.g.scale.z=m.h*ZS;m.mat.color.lerp(m.col,.12);m.mat.emissive.copy(m.mat.color).multiplyScalar(.45)})}
function loop(){requestAnimationFrame(()=>{tick();if(mode=='orbit'){C.position.set(rad*Math.sin(po)*Math.sin(az),rad*Math.cos(po),rad*Math.sin(po)*Math.cos(az));C.lookAt(0,0,0)}R.render(S,C);if(mode=='orbit')loop()})}
function startOrbit(){mode='orbit';holder.scale.setScalar(1);holder.position.set(0,0,0);holder.visible=true;S.background=null;R.setAnimationLoop(null);$('v').style.display='none';loop()}
// ---- AR cámara (giroscopio + vídeo): funciona en iOS/Android sin WebXR
$('cam').onclick=async()=>{try{
 if(window.DeviceOrientationEvent&&DeviceOrientationEvent.requestPermission)await DeviceOrientationEvent.requestPermission();
 const st=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});$('v').srcObject=st;$('v').style.display='block';$('v').play();
 mode='cam';holder.scale.setScalar(.7/EXT);holder.position.set(0,-.25,-1.4);C.position.set(0,0,0);C.rotation.order='YXZ';
 const q=new THREE.Quaternion(),e=new THREE.Euler(),q1=new THREE.Quaternion(-Math.sqrt(.5),0,0,Math.sqrt(.5)),d=Math.PI/180;
 addEventListener('deviceorientation',ev=>{if(mode!='cam'||ev.alpha==null)return;e.set(ev.beta*d,ev.alpha*d,-ev.gamma*d,'YXZ');C.quaternion.setFromEuler(e).multiply(q1)});
 R.setAnimationLoop(()=>{tick();R.render(S,C)})}catch(err){alert('AR cámara no disponible: '+err.message)}};
// ---- AR WebXR (hit-test): toque para anclar el mapa sobre una superficie (Chrome/Android + ARCore)
$('ar').onclick=async()=>{try{
 if(!navigator.xr||!(await navigator.xr.isSessionSupported('immersive-ar')))return alert('WebXR AR no disponible en este dispositivo. Use "AR cámara".');
 const s=await navigator.xr.requestSession('immersive-ar',{requiredFeatures:['hit-test']});R.xr.enabled=true;R.xr.setReferenceSpaceType('local');await R.xr.setSession(s);
 const vs=await s.requestReferenceSpace('viewer'),hs=await s.requestHitTestSource({space:vs});mode='xr';holder.visible=false;
 const ret=new THREE.Mesh(new THREE.RingGeometry(.07,.09,32).rotateX(-Math.PI/2),new THREE.MeshBasicMaterial({color:0x6dff5a}));ret.matrixAutoUpdate=false;ret.visible=false;S.add(ret);
 s.addEventListener('select',()=>{if(ret.visible){holder.position.setFromMatrixPosition(ret.matrix);holder.scale.setScalar(.5/EXT);holder.visible=true}});
 s.addEventListener('end',()=>{R.xr.enabled=false;S.remove(ret);startOrbit()});
 R.setAnimationLoop((t,f)=>{tick();if(f){const h=f.getHitTestResults(hs);if(h.length){ret.visible=!holder.visible||true;ret.matrix.fromArray(h[0].getPose(R.xr.getReferenceSpace()).transform.matrix)}else ret.visible=false}R.render(S,C)})
}catch(err){alert('WebXR: '+err.message)}};
useSet(Object.keys(D.sets)[0]);startOrbit();
</script>"""

CSS = """.gradio-container{background:#000!important;color:#6dff5a!important;font-family:'Courier New',monospace!important}
.gradio-container *{border-color:#1c5!important}h1,h2,h3,label,.prose,.prose *{color:#6dff5a!important}
button.primary{background:#0a3a14!important;color:#aaffaa!important;border:1px solid #6dff5a!important}
input,textarea,select,.dropdown,table,td,th{background:#031a08!important;color:#6dff5a!important}"""

# ============================================================ BLOQUE B — microdatos GEIH: descarga, variables, MCO
warnings.filterwarnings("ignore")

UA_MICRO = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Referer": "https://microdatos.dane.gov.co/"}

BASE = "https://microdatos.dane.gov.co/index.php/catalog/{cid}"

CATALOGOS = {2022: 771, 2023: 782, 2024: 819, 2026: 900}

KEYS = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]

COLS_CG = KEYS + ["MES", "DPTO", "AREA", "CLASE", "FEX_C18", "P3271", "P6040", "P3042"]

COLS_OC = KEYS + ["INGLABO", "P6800"]

ESC_NIVEL = {1: 0, 2: 0, 3: 5, 4: 9, 5: 11, 6: 11, 7: 13, 8: 13, 9: 14, 10: 16, 11: 17, 12: 18, 13: 22}

SEMANAS_POR_MES = 52 / 12

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())

def _session():
    import requests
    s = requests.Session(); s.headers.update(UA_MICRO); return s

def listar_archivos(catalog_id, ses=None):
    """Raspa «Obtener Microdatos» y devuelve [(resource_id, nombre_zip)] de los archivos mensuales."""
    ses = ses or _session()
    r = ses.get(BASE.format(cid=catalog_id) + "/get-microdata", timeout=60); r.raise_for_status()
    ids = list(dict.fromkeys(re.findall(r"/catalog/%d/download/(\d+)" % catalog_id, r.text)))
    nombres = re.findall(r"Nombre del archivo[^A-Za-z0-9]*([^<>|\n]+?\.zip)", r.text)
    if not ids: raise RuntimeError("No se hallaron enlaces de descarga en 'Obtener Microdatos' (¿cambió el portal o exige inicio de sesión?).")
    if len(nombres) != len(ids): nombres = [f"recurso_{i}.zip" for i in ids]          # el mes se detecta luego con la variable MES
    return list(zip(ids, nombres))

def descargar_mes(catalog_id, rid, nombre, outdir, ses):
    dest = Path(outdir) / f"{catalog_id}_{rid}_{_norm(nombre)}.zip"
    if dest.exists() and dest.stat().st_size > 10_000: return dest                       # reutiliza descargas previas
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = ses.get(BASE.format(cid=catalog_id) + f"/download/{rid}", timeout=300, allow_redirects=True); r.raise_for_status()
    if r.content[:2] != b"PK":
        raise RuntimeError(f"El portal no devolvió un .zip para {nombre} (devolvió HTML: posible aceptación de licencia/login). "
                           f"Descárguelo manualmente de {BASE.format(cid=catalog_id)}/get-microdata y colóquelo en {outdir}/")
    dest.write_bytes(r.content); return dest

def _leer_csv(zf, member, usecols):
    raw = zf.read(member)
    for enc in ("utf-8-sig", "latin-1"):
        try: txt = raw.decode(enc); break
        except UnicodeDecodeError: continue
    primera = txt.split("\n", 1)[0]
    sep = max([";", ",", "\t"], key=primera.count)                                       # el DANE publica con ';' (se detecta)
    df = pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, low_memory=False)
    df.columns = [c.strip().strip('"').upper() for c in df.columns]
    faltan = [c for c in usecols if c not in df.columns]
    if faltan: raise KeyError(f"{member}: faltan columnas oficiales {faltan}. Columnas disponibles (muestra): {list(df.columns)[:15]}")
    df = df[usecols].copy()
    for c in df.columns:                                                                  # numérico; admite coma decimal
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    return df

def leer_zip_mensual(path):
    """Une «Características generales» y «Ocupados» de un mes por (DIRECTORIO, SECUENCIA_P, ORDEN)."""
    with zipfile.ZipFile(path) as zf:
        miembros = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        cg = next((n for n in miembros if "caracteristicasgenerales" in _norm(Path(n).stem)), None)
        oc = next((n for n in miembros if _norm(Path(n).stem) == "ocupados"), None)       # excluye «No ocupados»
        if not (cg and oc): raise FileNotFoundError(f"{Path(path).name}: no se hallaron los CSV esperados. Contiene: {miembros}")
        a, b = _leer_csv(zf, cg, COLS_CG), _leer_csv(zf, oc, COLS_OC)
    for d in (a, b): d.drop_duplicates(KEYS, inplace=True)                                # llave única por persona-mes
    return b.merge(a, on=KEYS, how="inner", validate="one_to_one")

def cargar_geih(year=2024, meses=None, catalog_id=None, outdir="data/geih", verbose=True):
    """Descarga (o reutiliza) los zips mensuales del año y devuelve el DataFrame de ocupados con las variables oficiales."""
    cid = catalog_id or CATALOGOS.get(year)
    if cid is None: raise ValueError(f"Sin catálogo verificado para {year}. Conocidos: {CATALOGOS}. Pase catalog_id=… (número en la URL del ANDA).")
    outdir = Path(outdir) / str(year); ses = _session(); partes = []
    locales = sorted(outdir.glob("*.zip")) if outdir.exists() else []
    if not locales:
        for rid, nombre in listar_archivos(cid, ses):
            locales.append(descargar_mes(cid, rid, nombre, outdir, ses))
            if verbose: print(f"  ↓ {nombre}")
    for z in locales:
        try:
            m = leer_zip_mensual(z); partes.append(m)
            if verbose: print(f"  ✓ {z.name}: {len(m):,} ocupados")
        except Exception as ex: print(f"  ✗ {z.name}: {str(ex)[:150]}")
    if not partes: raise RuntimeError("No se pudo leer ningún mes.")
    df = pd.concat(partes, ignore_index=True)
    if meses is not None: df = df[df["MES"].isin(meses)]
    return df

def construir_muestra(df, edad_min=18, edad_max=65, trim=(0.01, 0.99)):
    """Devuelve (muestra, cascada). Cada filtro queda registrado para reportar la pérdida de observaciones."""
    c = []; d = df.copy(); c.append(("Ocupados (módulo Ocupados ⋈ Caract. generales)", len(d)))
    # (1) Ingreso laboral estrictamente positivo (INGLABO NaN/0 = sin ingreso o no informa)
    d = d[d["INGLABO"] > 0]; c.append(("INGLABO > 0", len(d)))
    # (2) Horas semanales válidas (P6800: 1–168 horas)
    d = d[d["P6800"].between(1, 168)]; c.append(("P6800 entre 1 y 168 h/semana", len(d)))
    # (3) Edad de trabajar (P6040)
    d = d[d["P6040"].between(edad_min, edad_max)]; c.append((f"Edad entre {edad_min} y {edad_max}", len(d)))
    # (4) Escolaridad válida: P3042 en 1–13 (99 = NS/NR → fuera)
    d = d[d["P3042"].isin(ESC_NIVEL)]; c.append(("P3042 válido (excluye 99/NaN)", len(d)))
    # Variables del modelo ----------------------------------------------------------------
    d["esc"] = d["P3042"].map(ESC_NIVEL).astype(float)                                   # años de escolaridad (construido)
    d["horas_mes"] = d["P6800"] * SEMANAS_POR_MES                                        # horas mensuales ≈ P6800 × 52/12
    d["w_hora"] = d["INGLABO"] / d["horas_mes"]                                          # ingreso laboral por hora (pesos corrientes)
    d["ln_w"] = np.log(d["w_hora"])
    d["exp"] = d["P6040"] - d["esc"] - 6                                                 # experiencia potencial = edad − esc − 6
    d["exp2"] = d["exp"] ** 2
    d["mujer"] = (d["P3271"] == 2).astype(int)                                           # ASUMIDO: 1 hombre, 2 mujer (ver docstring)
    d = d[d["exp"] >= 0]; c.append(("Experiencia potencial ≥ 0", len(d)))
    if trim:                                                                              # recorte de valores extremos de ln(w)
        lo, hi = d["ln_w"].quantile(list(trim)); d = d[d["ln_w"].between(lo, hi)]; c.append((f"ln(w) entre p{int(trim[0]*100)} y p{int(trim[1]*100)}", len(d)))
    d = d.dropna(subset=["ln_w", "esc", "exp", "DPTO", "FEX_C18"]); c.append(("Sin valores perdidos en variables del modelo", len(d)))
    d["DPTO"] = d["DPTO"].astype(int)
    return d, pd.DataFrame(c, columns=["filtro", "n"])

def estimar(d):
    """MCO con EE robustos. (1) Mincer clásico, HC1.  (2) + mujer + efectos fijos de departamento, EE clusterizados por DPTO.
       (3) igual a (2) por MCP con el factor de expansión FEX_C18 (sensibilidad)."""
    import statsmodels.formula.api as smf
    from statsmodels.iolib.summary2 import summary_col
    m1 = smf.ols("ln_w ~ esc + exp + exp2", d).fit(cov_type="HC1")
    m2 = smf.ols("ln_w ~ esc + exp + exp2 + mujer + C(DPTO)", d).fit(cov_type="cluster", cov_kwds={"groups": d["DPTO"]})
    m3 = smf.wls("ln_w ~ esc + exp + exp2 + mujer + C(DPTO)", d, weights=d["FEX_C18"]).fit(cov_type="cluster", cov_kwds={"groups": d["DPTO"]})
    tabla = summary_col([m1, m2, m3], stars=True, float_format="%.4f", model_names=["(1) Mincer HC1", "(2) +sexo +DPTO FE", "(3) MCP FEX_C18"],
                        regressors=["Intercept", "esc", "exp", "exp2", "mujer"],
                        info_dict={"N": lambda x: f"{int(x.nobs):,}", "R2": lambda x: f"{x.rsquared:.3f}"})
    return (m1, m2, m3), tabla

def interpretar(modelos):
    """Lectura básica de los β de la especificación (2)."""
    m = modelos[1]; b = m.params; pct = 100 * (np.exp(b["esc"]) - 1)
    L = [f"• Retorno a un año adicional de escolaridad: β_esc = {b['esc']:.4f} → ≈ {100*b['esc']:.1f}% (exacto: {pct:.1f}%) más ingreso/hora, ceteris paribus.",
         f"• Experiencia: β_exp = {b['exp']:.4f} (>0: el salario crece con la experiencia) y β_exp² = {b['exp2']:.5f} (<0: rendimientos decrecientes)."]
    if b["exp2"] < 0:
        pico = -b["exp"] / (2 * b["exp2"]); L.append(f"• El perfil experiencia–salario alcanza su máximo a ≈ {pico:.0f} años de experiencia potencial.")
    L.append(f"• Brecha por sexo: β_mujer = {b['mujer']:.4f} → las mujeres ganan ≈ {100*(np.exp(b['mujer'])-1):.1f}% por hora respecto a hombres con igual escolaridad, experiencia y departamento.")
    if b["mujer"] > 0: L.append("  ⚠️ Signo inesperado en 'mujer': verifique la codificación de P3271 en el formulario oficial (se asumió 1=hombre, 2=mujer).")
    L.append("• Advertencias: (i) esc es construida con duración nominal por nivel (sobrestima a quienes no terminaron el nivel → sesgo a la baja de β_esc); "
             "(ii) el retorno MCO no es causal (habilidad no observada, sesgo de selección: solo ocupados con ingreso>0); "
             "(iii) INGLABO es ingreso total mensual y P6800 solo horas del empleo principal → error de medición en w/hora.")
    return "\n".join(L)

def run(year=2024, meses=None, catalog_id=None, edad_min=18, edad_max=65, trim=(0.01, 0.99)):
    print(f"== GEIH {year}: descarga/lectura de microdatos oficiales (DANE-ANDA) ==")
    df = cargar_geih(year, meses, catalog_id)
    muestra, cascada = construir_muestra(df, edad_min, edad_max, trim)
    print("\n== Construcción de la muestra ==\n", cascada.to_string(index=False))
    print("\n== Estadísticos descriptivos ==\n", muestra[["w_hora", "ln_w", "esc", "P6040", "exp", "mujer"]].describe().round(2).to_string())
    modelos, tabla = estimar(muestra)
    print("\n== MCO de la ecuación de Mincer (EE robustos entre paréntesis; * p<.1 ** p<.05 *** p<.01) ==\n", tabla)
    print("\n== Interpretación ==\n" + interpretar(modelos))
    return dict(muestra=muestra, cascada=cascada, modelos=modelos, tabla=tabla)

# ============================================================ BLOQUE C — visualización, interpretación automática e interfaz
NIGHT = [[0, "#03140a"], [.35, "#0b7a2b"], [.7, "#5dff4a"], [1, "#eaffd8"]]

_CACHE = {}

def ols_hc1(y, X):
    """b = (X'X)^-1 X'y ;  V_HC1 = n/(n-k) (X'X)^-1 [Σ e_i² x_i x_i'] (X'X)^-1."""
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    Xe = X * e[:, None]; V = XtXi @ (Xe.T @ Xe * n / (n - k)) @ XtXi
    return b, np.sqrt(np.diag(V))

def _X(d): return np.column_stack([np.ones(len(d)), d["esc"], d["exp"], d["exp2"], d["mujer"]])

def estimar_departamentos(d, min_n=300):
    """Mincer por departamento (ln_w ~ esc + exp + exp2 + mujer, HC1). Departamentos con n<min_n: solo descriptivos."""
    rows = []
    for dp, g in d.groupby("DPTO"):
        w = g["FEX_C18"]
        r = dict(DPTO=int(dp), n=len(g), w_hora=np.average(g["w_hora"], weights=w), ln_w=np.average(g["ln_w"], weights=w),
                 esc=np.average(g["esc"], weights=w), ret=np.nan, ret_se=np.nan, brecha=np.nan)
        if len(g) >= min_n and g["esc"].nunique() > 2:
            try:
                b, se = ols_hc1(g["ln_w"].to_numpy(), _X(g)); r.update(ret=100 * b[1], ret_se=100 * se[1], brecha=100 * b[4])
            except np.linalg.LinAlgError: pass
        rows.append(r)
    return pd.DataFrame(rows)

def mes_por_depto(d):
    return d.groupby(["MES", "DPTO"]).apply(lambda g: np.average(g["ln_w"], weights=g["FEX_C18"])).unstack("DPTO")

def _layout3d(fig, titulo, ejes):
    ax = lambda t: dict(title=t, backgroundcolor="#000", gridcolor="#145a22", color="#6dff5a", showbackground=True)
    fig.update_layout(title=titulo, paper_bgcolor="#000", font=dict(color="#6dff5a", family="Courier New"), height=640,
                      margin=dict(l=0, r=0, t=40, b=0), scene=dict(xaxis=ax(ejes[0]), yaxis=ax(ejes[1]), zaxis=ax(ejes[2]), bgcolor="#000"))
    return fig

def fig_superficie(b, mujer_media):
    """Salario/hora predicho (miles de $) como función de escolaridad × experiencia, según el MCO nacional."""
    import plotly.graph_objects as go
    esc = np.linspace(0, 22, 45); ex = np.linspace(0, 45, 46); E, X = np.meshgrid(esc, ex)
    lnw = b[0] + b[1] * E + b[2] * X + b[3] * X ** 2 + b[4] * mujer_media
    f = go.Figure(go.Surface(x=E, y=X, z=np.exp(lnw) / 1000, colorscale=NIGHT, contours=dict(z=dict(show=True, color="#7cff6b", width=1)),
                             colorbar=dict(title="miles $/h", tickfont=dict(color="#6dff5a"))))
    return _layout3d(f, "Superficie de Mincer: salario/hora predicho", ("Escolaridad (años)", "Experiencia potencial (años)", "Salario/hora (miles $)"))

def fig_nube(d, n=4000, seed=0):
    import plotly.graph_objects as go
    s = d.sample(min(n, len(d)), random_state=seed); rng = np.random.default_rng(seed)
    f = go.Figure(go.Scatter3d(x=s["esc"] + rng.uniform(-.3, .3, len(s)), y=s["exp"], z=s["ln_w"], mode="markers",
                               marker=dict(size=2.2, color=s["ln_w"], colorscale=NIGHT, opacity=.75, colorbar=dict(title="ln(w)", tickfont=dict(color="#6dff5a"))),
                               text=[f"esc={e:.0f} exp={x:.0f} w/h=${w:,.0f}" for e, x, w in zip(s["esc"], s["exp"], s["w_hora"])], hoverinfo="text"))
    return _layout3d(f, f"Nube 3D de microdatos GEIH (muestra de {len(s):,} ocupados)", ("Escolaridad (años, con jitter)", "Experiencia potencial", "ln(salario/hora)"))

def fig_mapa2d(geo, vals, titulo, unidad):
    import plotly.graph_objects as go
    Ch = getattr(go, "Choroplethmap", None) or go.Choroplethmapbox
    f = go.Figure(Ch(geojson=geo["geojson"], locations=geo["keys"], z=vals, featureidkey="properties.key", text=geo["names"], colorscale=NIGHT,
                     marker_line_color="#7cff6b", marker_line_width=.6, hovertemplate="<b>%{text}</b><br>%{z:.2f} " + unidad + "<extra></extra>",
                     colorbar=dict(title=unidad, tickfont=dict(color="#6dff5a"))))
    lay = dict(center=dict(lat=4.6, lon=-73.5), zoom=4.3, style="carto-darkmatter")
    f.update_layout(**({"map": lay} if Ch.__name__ == "Choroplethmap" else {"mapbox": lay}), margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="#000", font=dict(color="#6dff5a"), height=640, title=titulo)
    return f

def interpretar_auto(nac, dep, W_info, meta):
    b, se = nac["b"], nac["se"]; L = []
    L.append("### Retorno nacional a la educación")
    L.append(f"- β_esc = **{b[1]:.4f}** (EE HC1 {se[1]:.4f}) → un año más de escolaridad se asocia con ≈ **{100*(np.exp(b[1])-1):.1f}%** más ingreso por hora (IC95 ≈ [{100*(np.exp(b[1]-1.96*se[1])-1):.1f}%, {100*(np.exp(b[1]+1.96*se[1])-1):.1f}%]).")
    if b[3] < 0: L.append(f"- Perfil experiencia–salario cóncavo (β_exp={b[2]:.4f}, β_exp²={b[3]:.5f}): máximo a ≈ **{-b[2]/(2*b[3]):.0f} años** de experiencia potencial.")
    L.append(f"- Brecha por sexo: β_mujer = {b[4]:.4f} → ≈ {100*(np.exp(b[4])-1):.1f}% por hora a igual escolaridad y experiencia." + ("  ⚠️ signo positivo: verifique la codificación de P3271." if b[4] > 0 else ""))
    v = dep.dropna(subset=["ret"]).sort_values("ret")
    L.append("### Heterogeneidad territorial")
    if len(v) >= 4:
        top, bot = v.iloc[-3:][::-1], v.iloc[:3]
        L.append(f"- Retornos departamentales (n={len(v)} deptos con ≥{meta['min_n']} obs.): rango **{v.ret.min():.1f}% – {v.ret.max():.1f}%** por año de escolaridad; desv. estándar {v.ret.std():.1f} pp.")
        L.append("- Mayores: " + ", ".join(f"{r.nombre} ({r.ret:.1f}%)" for r in top.itertuples()) + "  |  Menores: " + ", ".join(f"{r.nombre} ({r.ret:.1f}%)" for r in bot.itertuples()) + ".")
        hi, lo = v.iloc[-1], v.iloc[0]                              # departamento de mayor y de menor retorno
        distintos = (hi.ret - 1.96 * hi.ret_se) > (lo.ret + 1.96 * lo.ret_se)
        L.append("- " + ("Los extremos son estadísticamente distintos al 95% (sus IC no se solapan)." if distintos else "Los IC95 de los extremos **se solapan**: parte de la dispersión es ruido muestral, no diferencia real."))
        c = np.corrcoef(v.ret, v.esc)[0, 1]; L.append(f"- Correlación (entre deptos) retorno–escolaridad media = {c:.2f}: " + ("donde hay menos capital humano, cada año vale más (escasez relativa)." if c < -.3 else "donde hay más capital humano, cada año vale más (complementariedades)." if c > .3 else "sin relación clara."))
    if W_info and W_info.get("I") is not None:
        I, p = W_info["I"], W_info["p"]
        L.append(f"- Autocorrelación espacial de los retornos (Moran's I, W reina, n={W_info['n']}): **I = {I:.3f}** (p = {p:.3f}) → " + ("los retornos se **agrupan geográficamente** (departamentos vecinos se parecen)." if p < .05 and I > 0 else "patrón de competencia espacial (vecinos opuestos)." if p < .05 else "sin evidencia de agrupamiento espacial."))
    L.append("### Advertencias automáticas")
    W_ = [f"Muestra: {meta['n']:,} ocupados (año {meta['year']}). Años de escolaridad CONSTRUIDOS desde P3042 con duración nominal por nivel (sobrestima a quienes no terminaron el nivel → sesgo a la baja de β_esc).",
          "MCO no es causal: habilidad no observada y selección (solo ocupados con ingreso>0) sesgan el retorno; úselo como asociación condicional.",
          "Sexo: P3271 asumido 1=hombre, 2=mujer (el diccionario público solo da el rango 1–2).",
          "INGLABO es ingreso mensual total y P6800 solo horas del empleo principal → error de medición en el salario/hora.",
          f"Departamentos sin estimación propia (n<{meta['min_n']} o sin cobertura GEIH): {', '.join(meta['sin_est']) or 'ninguno'}."]
    return "\n".join(L) + "\n" + "\n".join("- " + w for w in W_)

def analizar(d, geo, year, min_n=300):
    n0 = len(geo["names"]); cod = {int(c): i for i, c in enumerate(geo["codigos"])}
    b, se = ols_hc1(d["ln_w"].to_numpy(), _X(d)); nac = dict(b=b, se=se)
    dep = estimar_departamentos(d, min_n); dep["idx"] = dep["DPTO"].map(cod); dep = dep.dropna(subset=["idx"]).copy(); dep["idx"] = dep["idx"].astype(int)
    dep["nombre"] = [geo["names"][i] for i in dep["idx"]]
    full = lambda col: [None if pd.isna(v) else float(v) for v in dep.set_index("idx")[col].reindex(range(n0))]
    meses = mes_por_depto(d); m_frames = []
    for mes in sorted(meses.index):
        row = meses.loc[mes]; v = np.full(n0, np.nan)
        for dp, val in row.items():
            if int(dp) in cod: v[cod[int(dp)]] = val
        m_frames.append([None if np.isnan(x) else float(x) for x in v])
    sets = {"ret": dict(label="Retorno a la educación por depto (% por año de escolaridad)", frames=["MCO por departamento"], vals=[full("ret")], unit="%"),
            "w": dict(label="Salario/hora medio (pesos)", frames=["promedio ponderado"], vals=[full("w_hora")], unit="$"),
            "esc": dict(label="Escolaridad media (años)", frames=["promedio ponderado"], vals=[full("esc")], unit="años"),
            "brecha": dict(label="Brecha salarial de mujeres por depto (%)", frames=["MCO por departamento"], vals=[full("brecha")], unit="%"),
            "mes": dict(label="ln(salario/hora) medio por mes (dinámica)", frames=[f"mes {int(m)}" for m in sorted(meses.index)], vals=m_frames, unit="ln $")}
    sets = {k: v for k, v in sets.items() if any(x is not None for fr in v["vals"] for x in fr)}
    # Moran sobre retornos (W reina con adyacencia MGN, solo deptos con estimación)
    W_info = None; ok = dep.dropna(subset=["ret"]); idx = ok["idx"].to_numpy()
    if len(idx) >= 8 and "adj" in geo:
        Wq = row_std(geo["adj"][np.ix_(idx, idx)].astype(float)); keep = Wq.sum(1) > 0
        if keep.sum() >= 8:
            Wk = row_std(geo["adj"][np.ix_(idx[keep], idx[keep])].astype(float)); I, p = moran(ok["ret"].to_numpy()[keep], Wk, perm=999)
            W_info = dict(I=I, p=p, n=int(keep.sum()))
    sin_est = sorted(set(geo["names"][i] for i in range(n0)) - set(ok["nombre"]))
    meta = dict(n=len(d), year=year, min_n=min_n, sin_est=sin_est)
    z = np.array([np.nan if v is None else v for v in full("ret")], float)
    return dict(html3d=scene_html(geo["geojson"], sets), fig_sup=fig_superficie(b, d["mujer"].mean()), fig_nube=fig_nube(d),
                fig_mapa=fig_mapa2d(geo, z, "Visión nocturna — retorno a un año de escolaridad", "%"), texto=interpretar_auto(nac, dep, W_info, meta),
                tabla_dep=dep[["nombre", "n", "ret", "ret_se", "brecha", "w_hora", "esc"]].round(3).sort_values("ret", ascending=False),
                tabla_nac=pd.DataFrame({"coef": b, "ee_HC1": se, "t": b / se}, index=["Intercepto", "esc", "exp", "exp2", "mujer"]).round(4))

def ejecutar(year, min_n=300):
    import gradio as gr
    try:
        key = ("muestra", int(year))
        if key not in _CACHE:
            df = cargar_geih(int(year)); _CACHE[key], _ = construir_muestra(df)
        if "geo" not in _CACHE: _CACHE["geo"] = load_geometry()
        r = analizar(_CACHE[key], _CACHE["geo"], int(year), int(min_n))
        return r["html3d"], r["fig_mapa"], r["fig_sup"], r["fig_nube"], r["texto"], r["tabla_dep"], r["tabla_nac"]
    except Exception as ex:
        raise gr.Error(f"{type(ex).__name__}: {str(ex)[:450]}")

def build_app():
    import gradio as gr
    kw = dict(css=CSS, title="Mincer Colombia — visión nocturna")
    try: blocks = gr.Blocks(head='<meta name="google" content="notranslate">', **kw)
    except TypeError: blocks = gr.Blocks(**kw)
    with blocks as app:
        gr.Markdown("# ▓ MINCER COLOMBIA — Retornos a la educación\n`ln(w) = β0 + β1·esc + β2·exp + β3·exp² + β4·mujer` · MCO con EE robustos HC1 · microdatos reales **GEIH-DANE** (marco 2018) · geometría MGN-DANE · sin datos simulados")
        with gr.Row():
            yr = gr.Dropdown([2022, 2023, 2024], value=2024, label="Año GEIH (catálogos verificados)")
            mn = gr.Slider(100, 1000, value=300, step=50, label="n mínimo para estimar por departamento")
            go_ = gr.Button("DESCARGAR Y ESTIMAR ▶", variant="primary")
        with gr.Tabs():
            with gr.Tab("3D · AR · dinámica"): h3 = gr.HTML()
            with gr.Tab("Mapa nocturno 2D"): pl = gr.Plot()
            with gr.Tab("Superficie 3D de Mincer"): sp = gr.Plot()
            with gr.Tab("Nube 3D de microdatos"): nb = gr.Plot()
            with gr.Tab("Interpretación automática"): md = gr.Markdown()
            with gr.Tab("Resultados"): t1 = gr.Dataframe(label="Por departamento"); t2 = gr.Dataframe(label="Nacional (MCO, HC1)")
        go_.click(ejecutar, [yr, mn], [h3, pl, sp, nb, md, t1, t2])
    return app

def main():
    _instalar()
    app = build_app()
    app.launch(share=True)
    return app


if __name__ == "__main__":
    main()
