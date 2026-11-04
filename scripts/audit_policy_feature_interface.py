"""Check merger-feature injection against native Qwen3-VL policy inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from tokenworld.evaluation.open_loop import ActionExpert,FidelityAccumulator
from tokenworld.data.dataset import RobotwinSVAETokenDataset

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnosis-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-frames",type=int,default=1)
    args=p.parse_args()
    cmd=json.loads(args.diagnosis_run.read_text())["command"]
    def get(key):return cmd[cmd.index("--"+key)+1]
    device=torch.device(args.device);torch.cuda.set_device(device)
    expert=ActionExpert(Path(get("policy-bundle")),"robotwin",device)
    policy=expert.policy;iface=policy.qwen_vl_interface;backbone=iface.model.model
    ds=RobotwinSVAETokenDataset(get("cache"),raw_data_path=get("raw-data-root"),split="val",
        num_hist=1,num_pred=1,views=[0],latent_method="symmetric_qwen")
    selected=json.loads(Path(get("selection")).read_text())["episodes"]
    tasks={"put_object_cabinet","turn_switch","shake_bottle_horizontally"}
    wanted={(r["dataset"],int(r["episode_index"])) for r in selected if r["task"] in tasks}
    rows=[];summaries=[]
    def metric(a,b):
        m=FidelityAccumulator();m.update(a,b);return m.report()
    for i,record in enumerate(ds.records):
        if (record["dataset"],int(record["episode_index"])) not in wanted:continue
        eid=f"{record['dataset']}_{int(record['episode_index']):06d}"
        saved_path=Path(get("output"))/(eid+".pt")
        saved=torch.load(saved_path,map_location="cpu",weights_only=True) if saved_path.exists() else None
        acc={label:{ref:FidelityAccumulator() for ref in ("native_full","merger_only")}
             for label in ("baseline","candidate")}
        frames=ds._load_raw_images(i,record,torch.arange(1,args.max_frames+1))
        for k,frame in enumerate(frames[:,0]):
            image=Image.fromarray(frame.numpy()).convert("RGB").resize((384,288),Image.BILINEAR)
            inputs=iface.build_qwenvl_inputs(images=[[image]],
                instructions=[expert.helper._action_instruction(policy,record["task"],record["outcome"])])
            inputs={key:v.to(device) for key,v in inputs.items() if torch.is_tensor(v)}
            merged,deep=backbone.get_image_features(inputs["pixel_values"],inputs["image_grid_thw"])
            feature=torch.stack(list(merged)).float().unsqueeze(0)
            template=expert.helper._build_policy_template(policy,record["task"],record["outcome"],(288,384))
            assert torch.equal(inputs["input_ids"],template["input_ids"].to(device))
            assert torch.equal(inputs["image_grid_thw"],template["image_grid_thw"].to(device))
            def native():
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    hidden=backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state
                queries=policy._gather_action_token_embeddings(hidden,inputs["input_ids"],action_token_id=policy.action_token_id)
                with torch.autocast("cuda",dtype=torch.float32):
                    return policy.action_model.predict_action(queries).float()
            full=native()
            original=backbone.get_image_features
            try:
                backbone.get_image_features=lambda *a,**kw:(merged,None)
                without_deepstack=native()
            finally:
                backbone.get_image_features=original
            injected=expert.helper._policy_actions(policy,template,feature)
            row={"episode_id":eid,"frame":k+1,"deepstack_levels":len(deep),
                 "injection_vs_native_without_deepstack":metric(without_deepstack,injected),
                 "merger_only_vs_full_native_policy":metric(full,injected)}
            if saved:
                row["saved_raw_teacher_vs_policy_visual_feature"]=metric(saved["raw_features"][k],feature.cpu())
                for label in acc:
                    pred=saved[label]["predicted_actions"][k]
                    acc[label]["native_full"].update(full,pred)
                    acc[label]["merger_only"].update(injected,pred)
            rows.append(row)
        summary={"episode_id":eid,"frames":args.max_frames,
                 "seed":3047,"metrics":{label:{ref:m.report() for ref,m in refs.items()}
                                      for label,refs in acc.items()}}
        summaries.append(summary);print(json.dumps(summary),flush=True)
    args.output.write_text(json.dumps({"status":"complete","rows":rows,"episode_summary":summaries,
        "note":"Native policy includes Qwen3 DeepStack; current fidelity protocol injects merger features only into both branches."},indent=2)+"\n")
if __name__=="__main__":main()
