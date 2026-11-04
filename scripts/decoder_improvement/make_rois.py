#!/usr/bin/env python3
"""Freeze GT-only object crop coordinates before candidate scoring."""
import argparse,json
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rois={}
    for c in torch.load(a.probes,map_location='cpu',weights_only=False):
        if c['split']=='train':continue
        rid=f"{c.get('dataset','seed')}_{c.get('episode_index',c.get('seed'))}_{c['frame']}"
        if c.get('seed')==100014:
            rgb=(F.interpolate(c['target'][None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]+1)/2
            mask=(rgb[0]>.55)&(rgb[0]-rgb[1]>.24)&(rgb[0]-rgb[2]>.24)
            todo=set(map(tuple,mask.nonzero().tolist()));largest=[]
            while todo:
                queue=deque([todo.pop()]);component=[]
                while queue:
                    point=queue.popleft();component.append(point);py,px=point
                    for neighbor in [(py-1,px),(py+1,px),(py,px-1),(py,px+1)]:
                        if neighbor in todo:todo.remove(neighbor);queue.append(neighbor)
                if len(component)>len(largest):largest=component
            coords=torch.tensor(largest);yy,xx=coords[:,0],coords[:,1]
            assert len(xx)>5,(rid,'red target not found')
            x=max(0,int(xx.min())-8);y=max(0,int(yy.min())-8);x1=min(224,int(xx.max())+9);y1=min(224,int(yy.max())+9)
            rois[rid]=[x,y,x1,y1]
        elif c['task']=='put_object_cabinet':rois[rid]=[45,0,185,190]
        elif c['task']=='click_alarmclock':rois[rid]=[45,45,115,205]
    a.output.write_text(json.dumps(rois,indent=2));print(json.dumps(rois),flush=True)
if __name__=='__main__':main()
