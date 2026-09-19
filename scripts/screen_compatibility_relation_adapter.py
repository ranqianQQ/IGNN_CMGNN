"""Validation-only capacity screen for a compatibility relation adapter."""
import argparse,copy,json,statistics,time
from pathlib import Path
import torch
from ignn.models import IGNN
from ignn.modules.CompatibilityRelationAdapter import CompatibilityRelationAdapter
from ignn.modules.compatibility_propagation import _row_normalized_adjacency,estimate_cmgnn_compatibility
from scripts.compatibility_experiment_common import seed_all
from scripts.screen_compatibility_finetuning import DATASETS,accuracy,masks_for,model_setup
CANDIDATES={'h8_frozen':dict(hidden=8,tune_classifier=False),'h16_frozen':dict(hidden=16,tune_classifier=False),'h32_frozen':dict(hidden=32,tune_classifier=False),'h16_joint':dict(hidden=16,tune_classifier=True)}
def fine_tune(model,embeddings,adj,data,masks,candidate,params,seed,max_epochs=300,patience=75):
 seed_all(seed);model.classifier.eval()
 with torch.no_grad():raw=model.classifier(embeddings);cm,diag=estimate_cmgnn_compatibility(raw,data.edge_index,data.y,masks[0])
 adapter=CompatibilityRelationAdapter(cm,hidden=candidate['hidden']).to(embeddings.device);groups=[{'params':adapter.parameters(),'lr':.01}]
 if candidate['tune_classifier']:groups.insert(0,{'params':model.classifier.parameters(),'lr':params['lr']*.1})
 else:
  for p in model.classifier.parameters():p.requires_grad_(False)
 optimizer=torch.optim.Adam(groups,weight_decay=params['l2_coef']);best_val,best_epoch,best_state=-1.,-1,None;history=[];started=time.perf_counter()
 for epoch in range(max_epochs):
  model.classifier.train(candidate['tune_classifier']);adapter.train();optimizer.zero_grad();raw=model.classifier(embeddings);corrected=adapter(raw,adj,data.y,masks[0]);loss=model.criterion(corrected[masks[0]],data.y[masks[0]])+1e-3*adapter.regularization()
  if candidate['tune_classifier']:loss=loss+.25*model.criterion(raw[masks[0]],data.y[masks[0]])
  loss.backward();optimizer.step();model.classifier.eval();adapter.eval()
  with torch.no_grad():val=accuracy(adapter(model.classifier(embeddings),adj,data.y,masks[0]),data.y,masks[1])
  history.append(val)
  if val>=best_val:best_val,best_epoch=val,epoch;best_state={'classifier':copy.deepcopy(model.classifier.state_dict()),'adapter':copy.deepcopy(adapter.state_dict())}
  if epoch-best_epoch>=patience:break
 model.classifier.load_state_dict(best_state['classifier']);adapter.load_state_dict(best_state['adapter'])
 return adapter,{'best_epoch':best_epoch+1,'best_val_accuracy':best_val,'epochs':epoch+1,'fine_tuning_seconds':time.perf_counter()-started,'matrix_shift_l2':float(adapter.matrix_residual.detach().norm()),'initial_cm_diagnostics':diag,'val_accuracy_history':history}
def main():
 p=argparse.ArgumentParser();p.add_argument('--split-end',type=int,default=3);p.add_argument('--max-epochs',type=int,default=300);p.add_argument('--datasets',nargs='+',default=list(DATASETS));p.add_argument('--output',default='experiments/compatibility_relation_adapter_screen.json');a=p.parse_args();device=torch.device('cuda:0');out=Path(a.output);result={'protocol':'global_validation_only_compatibility_relation_adapter','test_labels_evaluated':False,'candidates':CANDIDATES,'records':[]}
 for name in a.datasets:
  _,params,rn,data,config=model_setup(name,device);adj,_=_row_normalized_adjacency(data.edge_index,data.num_nodes,device)
  for split in range(a.split_end):
   masks=masks_for(data,split,device);seed=42+split
   for cname,candidate in CANDIDATES.items():
    seed_all(seed);model=IGNN(data.num_features,n_clusters=int(data.y.max())+1,IN='IN-SN',RN=rn,agg_type='gcn_incep',**params).to(device);model.load_state_dict(torch.load(f'experiments/sfd_10split_{name}_{split}_tuned_sfd.pt',map_location=device));model.eval()
    with torch.no_grad():embeddings=model(data.edge_index,data.x,config,device).detach();baseline=accuracy(model.classifier(embeddings),data.y,masks[1])
    adapter,fit=fine_tune(model,embeddings,adj,data,masks,candidate,params,seed,a.max_epochs);rec={'dataset':name,'split':split,'candidate':cname,'baseline_val_accuracy':baseline,'val_gain_pp':100*(fit['best_val_accuracy']-baseline),**fit};result['records'].append(rec);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print({k:v for k,v in rec.items() if k not in ('val_accuracy_history','initial_cm_diagnostics')},flush=True);del model,adapter,embeddings
 ranking=[]
 for cname in CANDIDATES:
  rows=[r for r in result['records'] if r['candidate']==cname];ranking.append({'candidate':cname,'mean_val_gain_pp':statistics.mean(r['val_gain_pp'] for r in rows),'positive_splits':sum(r['val_gain_pp']>1e-8 for r in rows),'total_splits':len(rows)})
 ranking.sort(key=lambda x:x['mean_val_gain_pp'],reverse=True);result['ranking']=ranking;result['selected_global_candidate']=ranking[0]['candidate'];out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(ranking,indent=2),flush=True)
if __name__=='__main__':main()
