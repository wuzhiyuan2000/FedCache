import os
import numpy as np
import utils
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import hnswlib
from scipy import spatial
import sys

np.random.seed(0)
def knowledge_avg(knowledge,weights):
    result=[]
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_,weights))
    return torch.Tensor(np.array(result)).cuda()

def knowledge_avg_single(knowledge,weights):
    result=torch.zeros_like(knowledge[0]).cpu()
    sum=0
    for _k,_w in zip(knowledge,weights):
        result.add_(_k.cpu()*_w)
        sum=sum+_w
    result=result/sum
    return torch.tensor(np.array(result.detach().cpu()))

class KnowledgeCache:
    def __init__(self,n_classes,R):
        self.n_classes=n_classes
        self.cache={}
        self.idx_to_hash={}
        self.relation={}
        for i in range(n_classes):
            self.cache[i]={}
        self.R=R
        pass

    def add_hash(self,hash,label,idx):
        for k_,l_,i_ in zip(hash,label,idx):
            self.add_hash_single(k_,l_,i_)

    def add_hash_single(self,hash,label,idx):
        self.cache[int(label)][idx]=torch.Tensor(np.array([0.0 for _ in range(self.n_classes)]))
        self.idx_to_hash[idx]=hash

    
    def build_relation(self):
        hnsw_sim = 0
        for c in range(self.n_classes):
            idx_vectors=[key for key in self.cache[c].keys()]
            data = list()
            data=np.array([self.idx_to_hash[key].numpy() for key in idx_vectors])
            num_elements = data.shape[0]
            dim = data.shape[1]
            data_labels = np.arange(num_elements)
            index = hnswlib.Index(space='cosine', dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=1000, M=64)
            index.add_items(data, data_labels)
            index.set_ef(1000)
            labels, distances = index.knn_query(data, self.R+1)
            for idx,ele in enumerate(labels):
                self.relation[idx_vectors[int(idx)]]=[]
                for x in ele[1:]:
                    self.relation[idx_vectors[int(idx)]].append(idx_vectors[x])

    def set_knowledge(self,knowledge,label,idx):
        for k_,l_,i_ in zip(knowledge,label,idx):
            self.set_knowledge_single(k_,l_,i_)

    def set_knowledge_single(self,knowledge,label,idx):
        self.cache[int(label)][idx]=knowledge

    def fetch_knowledge(self,label,idx):
        result=[]
        for l_,i_ in zip(label,idx):
            result.append(self.fetch_knowledge_single(l_,i_))
        return result

    def fetch_knowledge_single(self,label,idx):
        result=[]
        pairs=self.relation[idx]
        for pair in pairs:
            result.append(self.cache[int(label)][pair])
        return result

class FedCache_standalone_API:
    def __init__(self,client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args,test_data_global):
        self.client_models=client_models
        self.test_data_global=test_data_global
        self.global_logits_dict=dict()
        self.global_labels_dict=dict()
        self.global_extracted_feature_dict_test=dict()
        self.global_labels_dict_test=dict()
        self.criterion_KL = utils.KL_Loss()
        self.criterion_CE = F.cross_entropy
    

    def do_fedcache_stand_alone(self,client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
        image_scaler=transforms.Compose([
        transforms.Resize(224),
    ])
        print("*********start training with FedCache***************")
        train_data_local_dict_seq={}
        for client_index in range(args.client_number):
            train_data_local_dict_seq[client_index]=[]
            for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                train_data_local_dict_seq[client_index].append((images, labels))
        knowledge_cache=KnowledgeCache(args.class_num,args.R)
        encoder=mobilenet_v3_small(weights='IMAGENET1K_V1').cuda()
        encoder = torch.nn.Sequential( *( list(encoder.children())[:-1] ) )
        encoder.eval()
        for client_index,client_model in enumerate(self.client_models):
            cur_idx=0
            for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                images, labels=images.cuda(), labels.cuda()
                hash_code=encoder(image_scaler(images)).detach().cpu()
                hash_code=torch.tensor(hash_code.reshape((hash_code.shape[0],hash_code.shape[1])))
                for img,hash,label in zip(images,hash_code,labels):
                    knowledge_cache.add_hash_single(hash,label,(client_index,cur_idx))
                    cur_idx=cur_idx+1
        knowledge_cache.build_relation()
        print("*********knowledge cache initialized successfully***************")
        for global_epoch in range(args.comm_round):
            print("*********communication round",global_epoch,"***************")
            metrics_all={'test_loss':[],'test_accTop1':[],'test_accTop5':[],'f1':[]}
            for client_index,client_model in enumerate(self.client_models):
                client_model=self.client_models[client_index]
                print("*********start training on client",client_index,"***************")
                client_model=client_model.cuda()
                client_model.train()
                optim=torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                             weight_decay=args.wd)
                cur_idx=0
                for batch_idx, (images, labels) in enumerate(train_data_local_dict_seq[client_index]):
                    labels=torch.tensor(labels, dtype=torch.long)
                    images, labels = images.cuda(), labels.cuda()

                    log_probs = client_model(images)
                    loss_true = F.cross_entropy(log_probs, labels)
                    loss=None

                    teacher_knowledge=[]
                    for img,logit,label in zip(images,log_probs,labels):
                        fetched_knowledge_single=knowledge_cache.fetch_knowledge_single(label,(client_index,cur_idx))
                        knowledge_cache.set_knowledge_single(logit,label,(client_index,cur_idx))
                        cur_idx=cur_idx+1
                        avg_knowledge_single=knowledge_avg_single(fetched_knowledge_single,[1 for _ in range(args.R)])
                        teacher_knowledge.append(avg_knowledge_single.detach().cpu().numpy())
                    teacher_knowledge=torch.tensor(np.array(teacher_knowledge)).cuda()
                    loss_kd = self.criterion_KL(log_probs, teacher_knowledge/args.T)
                    loss = loss_true + args.alpha * loss_kd
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            if global_epoch%args.interval==0:
                acc_all=[]
                for client_index,client_model in enumerate(self.client_models):
                    if client_index%args.sel!=0:
                        continue
                    print("*********start tesing on client",client_index,"***************")
                    client_model.eval()
                    loss_avg = utils.RunningAverage()
                    accTop1_avg = utils.RunningAverage()
                    accTop5_avg = utils.RunningAverage()
                    for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                        images, labels = images.cuda(), labels.cuda()
                        labels=torch.tensor(labels,dtype=torch.long)
                        log_probs = client_model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        accTop1_avg.update(metrics[0].item())
                        accTop5_avg.update(metrics[1].item())
                        loss_avg.update(loss.item())
                    test_metrics = {str(client_index)+' test_loss': loss_avg.value(),
                                    str(client_index)+' test_accTop1': accTop1_avg.value(),
                                    str(client_index)+' test_accTop5': accTop5_avg.value(),
                                    }
                    acc=accTop1_avg.value()
                    print("mean Test/AccTop1 on client",client_index,":",acc)
                    acc_all.append(acc)
                print("mean Test/AccTop1 on all clients:",float(np.mean(np.array(acc_all))))

