"""Deterministic software checks for the video2xy manuscript.

Usage: python experiments.py --repo /path/to/video2xy --out results
No camera, sound device, or external dataset is used.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.repo.resolve()))
import video_to_xy_audio as v

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
metrics = {'commit': subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip(),
           'python': platform.python_version(), 'platform': platform.platform(),
           'numpy': np.__version__, 'opencv': cv2.__version__, 'matplotlib': matplotlib.__version__}

def box(x,y,w,h):
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)

def build(cs, preserve):
    return v.build_trace(cs, 320, 240, 735, 1.25, 0.8, 1, 8, preserve_order=preserve)

def rmse(x,y):
    return float(np.sqrt(np.mean((x.astype(np.float64)-y.astype(np.float64))**2)))

# Deliberately separate shape identity from its list representation.
base = [box(20,25,65,45),box(160,120,45,30),box(240,40,25,20)]
tracker = v.ContourTracker()
tracked_ref = build(tracker.update(base,320,240), True)
untracked_ref = build(base,False)
rng = np.random.default_rng(20260913)
rows=[]
leading_rows=[]
tracked_exact_matches=0
for k in range(200):
    altered=[]
    for c in base:
        c=c[::-1] if rng.integers(0,2) else c.copy()
        c=np.roll(c,int(rng.integers(0,len(c))),axis=0)
        altered.append(c)
    phase_only=build(altered,False)
    permutation=rng.permutation(len(altered))
    altered=[altered[i] for i in permutation]
    t=build(tracker.update(altered,320,240),True)
    u=build(altered,False)
    rows.append([k+1,rmse(t,tracked_ref),rmse(u,untracked_ref),rmse(phase_only,untracked_ref)])
    leading_rows.append([k+1,int(permutation[0])+1])
    tracked_exact_matches+=int(np.array_equal(t,tracked_ref))
with (a.out/'permutation.csv').open('w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['trial','tracked_rmse','untracked_rmse','phase_direction_only_untracked_rmse']); wr.writerows(rows)
arr=np.array(rows)
with (a.out/'leading_contour.csv').open('w',newline='') as f:
    wr=csv.writer(f); wr.writerow(['trial','leading_contour_id']); wr.writerows(leading_rows)
leaders=np.array(leading_rows)[:,1]
metrics['permutation']={'trials':len(rows),'seed':20260913,
    'tracked_mean_rmse':float(arr[:,1].mean()),'tracked_max_rmse':float(arr[:,1].max()),
    'untracked_mean_rmse':float(arr[:,2].mean()),'untracked_max_rmse':float(arr[:,2].max()),
    'untracked_nonzero_trials':int(np.count_nonzero(arr[:,2]>1e-6)),
    'phase_direction_only_untracked_mean_rmse':float(arr[:,3].mean()),
    'phase_direction_only_untracked_max_rmse':float(arr[:,3].max())}
def summarize_errors(values):
    changed=values[values>1e-6]
    return {'trials':len(values),'changed_trials':len(changed),
            'minimum_rmse':float(values.min()),'maximum_rmse':float(values.max()),
            'changed_minimum_rmse':float(changed.min()) if len(changed) else None,
            'changed_maximum_rmse':float(changed.max()) if len(changed) else None}

metrics['representation_summary']={
    'change_threshold':1e-6,'tracked_exact_matches':tracked_exact_matches,
    'tracked':summarize_errors(arr[:,1]),
    'fully_permuted_untracked':summarize_errors(arr[:,2]),
    'fixed_order_untracked':summarize_errors(arr[:,3]),
    'by_leading_contour':[
        {'contour_id':i,'perimeter_px':v.polyline_length(base[i-1]),
         **summarize_errors(arr[leaders==i,2])} for i in (1,2,3)]}

# Conditional strip plot: horizontal position is measured error; vertical
# offsets separate overlapping markers and have no numerical interpretation.
fig,ax=plt.subplots(figsize=(6.4,3.45),layout='constrained')
groups=[(arr[:,1],'Tracking; all changes (n=200)','#005c87'),
        (arr[:,3],'Untracked; fixed list (n=200)','#747474')]
for i,color in zip((1,2,3),('#9d572a','#744c9d','#238070')):
    values=arr[leaders==i,2]
    groups.append((values,f'Untracked; lead {i} (n={len(values)})',color))
for j,(values,label,color) in enumerate(groups):
    offsets=.19*np.sin(np.arange(len(values))*2.399963229728653)
    ax.scatter(values,j+offsets,s=9,alpha=.6,color=color,edgecolors='none')
ax.set_yticks(range(len(groups)),[g[1] for g in groups],fontsize=8)
ax.set(xlabel='Trace RMSE (digital units)',xlim=(-.02,.71),ylim=(4.5,-.55))
ax.axhline(1.5,color='#bbbbbb',lw=.6)
ax.grid(axis='x',color='#dddddd',lw=.5)
ax.set_axisbelow(True)
fig.savefig(a.out/'tracking.pdf'); fig.savefig(a.out/'tracking.png',dpi=180); plt.close(fig)

# Motion and subdivision follow the repository's test construction.
tracker=v.ContourTracker()
rect=box(20,20,30,20)
first=tracker.update([rect],320,240)[0]
subdiv=np.insert(rect,1,(rect[0]+rect[1])/2,axis=0)
moved=tracker.update([np.roll(subdiv[::-1]+[3,2],2,axis=0)],320,240)[0]
metrics['translation']={'displacement_px':[3,2],
    'start_error_px':float(np.max(np.abs(moved[0]-(first[0]+[3,2])))),
    'trace_rmse':rmse(build([moved],True),build([first+[3,2]],True))}

# Accumulated rounding compared to exact rational time at every prefix.
timing=[]
for fps in [Fraction(24),Fraction(25),Fraction(30),Fraction(30000,1001)]:
    clock=v.FrameSampleClock(44100,float(fps))
    total=0; maxerr=Fraction(0); unique=set()
    for k in range(1,100001):
        n=clock.next_count(); unique.add(n); total+=n
        maxerr=max(maxerr,abs(Fraction(total)-Fraction(k*44100,1)/fps))
    timing.append({'fps':str(fps),'frames':100000,'frame_sample_counts':sorted(unique),
                   'max_prefix_error_samples':float(maxerr),'total_samples':total,
                   'final_error_samples':float(Fraction(total)-Fraction(100000*44100,1)/fps)})
metrics['timing']=timing

# A deterministic raster example; show ideal sample geometry, not a CRT simulation.
frame=np.zeros((240,320,3),np.uint8)
cv2.rectangle(frame,(24,30),(116,102),(255,255,255),-1)
cv2.circle(frame,(233,66),33,(255,255,255),-1)
cv2.fillPoly(frame,[np.array([[123,157],[187,209],[69,209]],np.int32)],(255,255,255))
processed,edges=v.preprocess_edges(frame,480,1.5,5,3,72,35,110,1)
contours=v.extract_contours(edges,18,32,.004)
contours=v.ContourTracker().update(contours,480,360)
trace=v.build_trace(contours,480,360,735,1.25,.8,1,8,preserve_order=True)
lengths=np.array([v.polyline_length(c) for c in contours])
counts=v.allocate_samples(lengths,735-len(contours),8)
metrics['raster']={'input_size':[320,240],'processed_size':[480,360],
    'edge_pixels':int(np.count_nonzero(edges)),'contours':len(contours),
    'vertices':[len(c) for c in contours], 'counts':[int(n) for n in counts],
    'n_trace':len(trace),'max_abs':float(np.max(np.abs(trace)))}
fig,axs=plt.subplots(1,3,figsize=(6.45,2.3),layout='constrained')
axs[0].imshow(cv2.cvtColor(processed,cv2.COLOR_BGR2RGB));axs[0].axis('off');axs[0].set_title('(a) Synthetic input',fontsize=9)
axs[1].imshow(edges,cmap='gray');axs[1].axis('off');axs[1].set_title('(b) Selected edges',fontsize=9)
ax=axs[2];off=0
for i,n in enumerate(counts):
    t=trace[off:off+n]
    ax.plot(t[:,0],t[:,1],color='#005c87',lw=.65)
    j=trace[off+n-1:off+n+1]
    ax.plot(j[:,0],j[:,1],color='#ac4835',lw=.7,linestyle='--')
    off+=n+1
ax.set(xlim=(-.85,.85),ylim=(-.85,.85),xlabel='X',ylabel='Y')
ax.set_aspect(1/1.25);ax.set_title('(c) Sample trajectory',fontsize=9)
ax.set_xticks([-.8,0,.8]);ax.set_yticks([-.8,0,.8])
fig.savefig(a.out/'pipeline_example.pdf');fig.savefig(a.out/'pipeline_example.png',dpi=180);plt.close(fig)

# Repeat the upstream suite and preserve stdout/stderr as actual evidence.
run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.repo),'-v'],capture_output=True,text=True)
(a.out/'upstream_tests.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
metrics['upstream_test_exit_code']=run.returncode
if run.returncode: raise RuntimeError(run.stderr)
(a.out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
print(json.dumps(metrics,indent=2))
