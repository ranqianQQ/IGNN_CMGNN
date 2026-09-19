"""Ten-split evaluation of the globally selected compatibility relation adapter."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
from ignn.models import IGNN
from ignn.modules.compatibility_propagation import _row_normalized_adjacency
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_compatibility_finetuning import DATASETS,accuracy,masks_for,model_setup
from scripts.screen_compatibility_relation_adapter import fine_tune

SELECTED=dict(hidden=16,tune_classifier=False)
def mean_std(x):return {'mean':statistics.mean(x),'std':statistics.stdev(x)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=list(DATASETS));p.add_argument('--max-epochs',type=int,default=300);p.add_argument('--output',default='experiments/compatibility_relation_adapter_10splits.json');a=p.parse_args();device=torch.device('cuda:0');torch.set_num_threads(4);out=Path(a.output);result={'protocol':'fixed_global_configuration_official_10_splits','selection':'hidden=16 frozen classifier selected by mean validation gain on splits 0-2 across all eight datasets; test unseen','selected':SELECTED,'records':[]}
 for name in a.datasets:
  _,params,rn,data,config=model_setup(name,device);adj,_=_row_normalized_adjacency(data.edge_index,data.num_nodes,device)
  for split in range(10):
   masks=masks_for(data,split,device);seed=42+split;seed_all(seed);model=IGNN(data.num_features,n_clusters=int(data.y.max())+1,IN='IN-SN',RN=rn,agg_type='gcn_incep',**params).to(device);warm=Path(f'experiments/sfd_10split_{name}_{split}_tuned_sfd.pt');model.load_state_dict(torch.load(warm,map_location=device));model.eval()
   with torch.no_grad():embeddings=model(data.edge_index,data.x,config,device).detach();raw=model.classifier(embeddings);base_val=accuracy(raw,data.y,masks[1]);base_test=accuracy(raw,data.y,masks[2])
   adapter,fit=fine_tune(model,embeddings,adj,data,masks,SELECTED,params,seed,a.max_epochs);model.classifier.eval();adapter.eval()
   with torch.no_grad():scores=adapter(model.classifier(embeddings),adj,data.y,masks[0]);val=accuracy(scores,data.y,masks[1]);test=accuracy(scores,data.y,masks[2]);cm=adapter.compatibility();row_error=float((cm.sum(1)-1).abs().max())
   checkpoint=out.parent/f'cmra_10split_{name}_{split}.pt';torch.save({'adapter':adapter.state_dict(),'warmup_checkpoint':warm.name,'selected':SELECTED},checkpoint)
   rec={'dataset':name,'split':split,'seed':seed,'split_hash':hashlib.sha256(b''.join(m.cpu().numpy().tobytes() for m in masks)).hexdigest(),'warmup_checkpoint':warm.name,'checkpoint':checkpoint.name,'baseline_val_accuracy':base_val,'best_val_accuracy':val,'val_gain_pp':100*(val-base_val),'baseline_test_accuracy':base_test,'test_accuracy':test,'test_gain_pp':100*(test-base_test),'adapter_parameters':sum(p.numel() for p in adapter.parameters()),'compatibility_row_sum_max_error':row_error,**fit};result['records'].append(rec);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in rec.items() if k not in ('val_accuracy_history','initial_cm_diagnostics')},flush=True);del model,adapter,embeddings
 summary={}
 for name in a.datasets:
  rows=sorted((r for r in result['records'] if r['dataset']==name),key=lambda r:r['split']);g=[r['test_gain_pp'] for r in rows];summary[name]={'baseline_test':mean_std([100*r['baseline_test_accuracy'] for r in rows]),'relation_adapter_test':mean_std([100*r['test_accuracy'] for r in rows]),'paired_test_gain_pp':mean_std(g),'wins_ties_losses':[sum(x>1e-8 for x in g),sum(abs(x)<=1e-8 for x in g),sum(x< -1e-8 for x in g)],'mean_val_gain_pp':statistics.mean(r['val_gain_pp'] for r in rows),'mean_fine_tuning_seconds':statistics.mean(r['fine_tuning_seconds'] for r in rows)}
 result['summary']=summary;result['macro_average_paired_test_gain_pp']=statistics.mean(v['paired_test_gain_pp']['mean'] for v in summary.values());out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'summary':summary,'macro_average_paired_test_gain_pp':result['macro_average_paired_test_gain_pp']},indent=2),flush=True)
if __name__=='__main__':main()
