import requests,tarfile,io,pathlib,zipfile,zstandard
root=pathlib.Path('/nobackup2/le/.cache/cuda-m2t2-12.6');root.mkdir(exist_ok=True)
queue=['cuda-nvcc-tools','cuda-nvcc-dev_linux-64','cuda-nvvm-impl','cuda-cudart-dev_linux-64'];done=set()
while queue:
 name=queue.pop(0)
 if name in done or name=='cuda-version':continue
 done.add(name)
 j=requests.get('https://api.anaconda.org/package/nvidia/'+name,timeout=60).json()
 xs=[x for x in j.get('files',[]) if x['version'].startswith('12.6') and x['basename'].startswith(('linux-64','noarch'))]
 if not xs:print('SKIP',name,flush=True);continue
 x=xs[-1];print('Downloading',x['basename'],flush=True)
 queue += [s.split()[0] for s in x.get('attrs',{}).get('depends',[]) if s.startswith('cuda-')]
 r=requests.get('https:'+x['download_url'],timeout=120);r.raise_for_status()
 if x['basename'].endswith('.conda'):
  z=zipfile.ZipFile(io.BytesIO(r.content));fn=next(n for n in z.namelist() if n.startswith('pkg-'))
  with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(fn))) as stream:
   with tarfile.open(fileobj=stream,mode='r|') as t:t.extractall(root,filter='data')
 else:
  with tarfile.open(fileobj=io.BytesIO(r.content),mode='r:bz2') as t:t.extractall(root,filter='data')
