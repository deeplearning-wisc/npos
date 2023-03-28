from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.cuda()
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def get_Mahalanobis_score(args, model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, num_batches, in_dist=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break
        
        data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, requires_grad = True), Variable(target)
        
        # out_features = model.encoder.intermediate_forward(data, layer_index)
        # out_features = model.intermediate_forward(data, layer_index)
        
        # out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        # out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        # gaussian_score = 0
        # for i in range(num_classes):
        #     batch_sample_mean = sample_mean[layer_index][i]
        #     zero_f = out_features.data - batch_sample_mean
        #     term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        #     if i == 0:
        #         gaussian_score = term_gau.view(-1,1)
        #     else:
        #         gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        # sample_pred = gaussian_score.max(1)[1]
        # batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        # zero_f = out_features - Variable(batch_sample_mean)
        # pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        # loss = torch.mean(-pure_gau)
        # loss.backward()
         
        # gradient =  torch.ge(data.grad.data, 0)
        # gradient = (gradient.float() - 0.5) * 2
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        # tempInputs = torch.add(data.data, -magnitude, gradient)
        tempInputs = data.data
        with torch.no_grad():
            # noise_out_features = model.encoder.intermediate_forward(tempInputs, layer_index)
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        #DEBUG NORM
        if args.norm_pe:
            if layer_index == 0:
                 noise_out_features = F.normalize(noise_out_features, dim = 1)
        #END
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32)

def sample_estimator(args, model, num_classes, feature_list, train_loader, head = False):
    """
    compute sample mean and precision (inverse of covariance)
    feature_list: array([ 64.,  64., 128., 256., 512.]) (resnet18 as an example)
    return: sample_class_mean: list of class mean
             precision: list of precisions

    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list) # 5
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = [] # dim (5, 10)
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list) 
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        # data = Variable(data, volatile=True)
        out_features = model.encoder.feature_list(data)

        output = model(data)
        # output, out_features = model.feature_list(data)
        # DEBUG REF
        # out_features[0].shape
        # torch.Size([64, 64, 32, 32])
        # out_features[1].shape
        # torch.Size([64, 64, 32, 32])
        # out_features[2].shape
        # torch.Size([64, 128, 16, 16])
        # out_features[3].shape
        # torch.Size([64, 256, 8, 8])
        # out_features[4].shape
        # torch.Size([64, 512, 4, 4])
        #END
        # get hidden features
        out_features[0] = out_features[0].view(out_features[0].size(0), out_features[0].size(1), -1)
        out_features[0] = torch.mean(out_features[0].data, 2)  #after this: out_features[0].shape 64, 512
        #DEBUG NORM
        if args.norm_pe:
            out_features[0] = F.normalize(out_features[0], dim = 1)
        #END

        #DEBUG REF
        # out_features[0].shape
        # torch.Size([64, 64])
        # out_features[1].shape
        # torch.Size([64, 64])
        # out_features[2].shape
        # torch.Size([64, 128])
        # out_features[3].shape
        # torch.Size([64, 256])
        # out_features[4].shape
        # torch.Size([64, 512])
        #END
        # compute the accuracy
        pred = output.data.max(1)[1] #shape: [64] tensor([6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6, 4, 3, 6, 6, ...])
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list: #array([512.])
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
    #DEBUG REF
    # sample_class_mean[0].shape
    # torch.Size([10, 64])
    # sample_class_mean[1].shape
    # torch.Size([10, 64])
    # sample_class_mean[2].shape
    # torch.Size([10, 128])
    # sample_class_mean[3].shape
    # torch.Size([10, 256])
    # sample_class_mean[4].shape
    # torch.Size([10, 512])      
    #END
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision
    #sample_class_mean[0].shape: 10, 512
    # precision[0].shape: 512, 512


def sample_estimator_with_head(log, model, num_classes, feature_list, train_loader, multiplier = 1):
    """
    compute sample mean and precision (inverse of covariance)
    feature_list: array([ 64.,  64., 128., 256., 512.]) (resnet18 as an example)
    return: sample_class_mean: list of class mean
             precision: list of precisions

    """
    # import sklearn.covariance
    
    model.eval()
    with torch.no_grad():
        # group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = [] # dim (5, 10)
        num_output = len(feature_list) 
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list) 
        
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            # data = Variable(data, volatile=True)

            #original 
            out_features = model.encoder.feature_list(data)
            out_features[0] = out_features[0].view(out_features[0].size(0), out_features[0].size(1), -1)
            out_features[0] = torch.mean(out_features[0].data, 2)  
            # now out_features[0] shape: bsz, 512 (last conv layer feature dim)
            feat = model.encoder(data)
            output = model.fc(feat)
            #new
            head = multiplier * F.normalize(model.head(feat), dim=1)
            #old
             #head = multiplier * F.normalize(model.head(out_features[-1]), dim=1)
            ##for ablation a
            # head = multiplier * F.normalize(model.head(feat), dim=1)
            ##head = F.normalize(out_features[-1], dim=1)
            
            out_features.append(head)  
            #end original 
            
            # compute the accuracy
            pred = output.data.max(1)[1] #shape: [64] tensor([6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6, 4, 3, 6, 6, ...])
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1
                
        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:  #feature_list = array([512., 128.]) for resnet18
            temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1
            
        precision = []
        for k in range(num_output): #num_output = 2
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
            # # find inverse            
            # group_lasso.fit(X.cpu().numpy())
            # temp_precision = group_lasso.precision_
            # # vals = np.linalg.eigvals(temp_precision)
            # # print(f"before: max eigen val {np.max(vals)} min eigen val {np.min(vals)}")
            # temp_precision = torch.from_numpy(temp_precision).float().cuda()
            #attempt 1
            # import random
            # subset_idx = random.sample(range(len(X)), 5000)
            # X = X[subset_idx]
            #end

            cov = torch.cov(X.T.double())
            with open('cov.npy', 'wb') as f:
                np.save(f,cov.cpu().numpy())
            #attempt 2
            cov = cov + 1e-7*torch.eye(X.shape[1]).cuda()
            #end
            log.debug(f'k = {k}: cov of X\n {cov}')
            vals = torch.eig(cov)[0]
            log.debug(f"before inv: max eigen val {torch.max(vals)} min eigen val {torch.min(vals)}")
            log.debug(f'cond number: {torch.linalg.cond(cov)}')
            # temp_precision = torch.linalg.pinv(cov) #not used
            # temp_precision = torch.linalg.inv(cov.cpu())
            temp_precision = torch.linalg.inv(cov)
            log.debug(f'k = {k}: inverse cov of X\n {temp_precision}')
            vals = torch.eig(temp_precision)[0]
            log.debug(f"after inv: max eigen val {torch.max(vals)} min eigen val {torch.min(vals)}")
            log.debug(f'cond number: {torch.linalg.cond(temp_precision)}')

            # attempt 3
            # from sklearn.covariance import GraphicalLassoCV
            # approximator = GraphicalLassoCV()
            # approximator.fit(X.cpu().numpy())
            # temp_precision = approximator.precision_
            # temp_precision = torch.from_numpy(temp_precision).cuda()
            # log.debug(f'cond number: {torch.linalg.cond(temp_precision)}')
            #end
            precision.append(temp_precision.float())
            
        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
        with open('mean.npy', 'wb') as f:
            np.save(f,sample_class_mean[1].cpu().numpy())
        return sample_class_mean, precision


def mean_estimator_with_head_original(model, num_classes, num_feature, train_loader):
    """
    the version submitted to CVPR
    """
    model.eval()
    with torch.no_grad():
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = [0]* num_classes 

        for data, target in train_loader:
            data = data.cuda()
            data = Variable(data, volatile=True)
            feat = model.encoder(data)
            head = F.normalize(model.head(feat), dim=1) 

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    list_features[label] = head[i].view(1, -1)
                else:
                    list_features[label] \
                        = torch.cat((list_features[label], head[i].view(1, -1)), 0)            
                num_sample_per_class[label] += 1
                
        sample_class_mean = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            sample_class_mean[j] = torch.mean(list_features[j], 0)
  
        return sample_class_mean

def mean_estimator_with_head(model, num_classes, num_feature, train_loader, layer_idx):
    """
    new version Dec 6
    """
    model.eval()
    with torch.no_grad():
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = [0]* num_classes 

        for data, target in train_loader:
            data = data.cuda()
            data = Variable(data, volatile=True)
            feat = model.encoder(data)
            if layer_idx == 0:
                head = F.normalize(feat, dim=1) 
            else:
                head = F.normalize(model.head(feat), dim=1) 

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    list_features[label] = head[i].view(1, -1)
                else:
                    list_features[label] \
                        = torch.cat((list_features[label], head[i].view(1, -1)), 0)            
                num_sample_per_class[label] += 1 # not used later
                
        sample_class_mean = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            #old
            # sample_class_mean[j] = torch.mean(list_features[j], 0)
            #new 
            sample_class_mean[j] = torch.sum(list_features[j], 0)
            sample_class_mean[j] = sample_class_mean[j] / torch.linalg.vector_norm(sample_class_mean[j])
        return sample_class_mean

def mean_estimator_without_head(model, num_classes, num_feature, train_loader):
    """
    """
    model.eval()
    with torch.no_grad():
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = [0]* num_classes 

        for data, target in train_loader:
            data = data.cuda()
            data = Variable(data, volatile=True)
            feat = model.encoder(data)
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    list_features[label] = feat[i].view(1, -1)
                else:
                    list_features[label] \
                        = torch.cat((list_features[label], feat[i].view(1, -1)), 0)            
                num_sample_per_class[label] += 1
                
        sample_class_mean = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            sample_class_mean[j] = torch.mean(list_features[j], 0)
  
        return sample_class_mean

def get_cosine_score(log, model, test_loader, num_classes, sample_mean, num_batches, layer_index, in_dist = False):
    '''
    sample mean. shape: [10, 128]
    '''
    model.eval()
    cosine = []
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            out_features = model.intermediate_forward(data, layer_index)     # 64, 128
            if layer_index == 0:
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                out_features = torch.mean(out_features, 2)
                out_features =F.normalize(out_features, dim = 1)
        # compute cosine similarity score
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i] #sample_mean shape: # of cls, 128
            if i == 0:
                score = torch.mm(out_features, batch_sample_mean.view(-1,1))
            else:
                score = torch.cat((score, torch.mm(out_features, batch_sample_mean.view(-1,1))), 1)   

        score, _ = torch.max(score, dim=1) #values, indices
        cosine.extend(-score.cpu().numpy())
        # cosine.extend(2-2*score.cpu().numpy())
    #DEBUG
    if in_dist:
        avg_dist = np.zeros(num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if j!= i: 
                    cos_score = torch.dot(sample_mean[i], sample_mean[j])
                    degree = np.degrees(np.arccos(cos_score.detach().cpu().numpy()))
                    avg_dist[i] += degree 
        avg_dist = avg_dist /(num_classes-1)
        log.debug(f'Avg degree between mean vectors: {avg_dist}; ON avg: {avg_dist.mean()}')

    #DEBUG
        # all_prod = np.zeros((num_classes, num_classes))
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         cos_score = torch.dot(sample_mean[i], sample_mean[j])
        #         degree = np.degrees(np.arccos(cos_score.detach().cpu().numpy()))
        #         all_prod[i,j] = degree
        # log.debug(f'degree among all mean vectors: {all_prod}')
    return np.asarray(cosine, dtype=np.float32)

def get_cosine_score_pe(model, test_loader, num_classes, sample_mean, num_batches, layer_index):
    '''
    sample mean. shape: [10, 128]
    '''
    model.eval()
    cosine = []
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sample_mean = F.normalize(sample_mean, dim = 1)
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            out_features = model.intermediate_forward(data, layer_index)     # 64, 128
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            out_features =F.normalize(out_features, dim = 1)
        # compute cosine similarity score
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i] #sample_mean shape: # of cls, 128
            if i == 0:
                score = torch.mm(out_features, batch_sample_mean.view(-1,1))
            else:
                score = torch.cat((score, torch.mm(out_features, batch_sample_mean.view(-1,1))), 1)   

        score, _ = torch.max(score, dim=1) #values, indices
        cosine.extend(score.cpu().numpy())
    cosine = np.asarray(cosine, dtype=np.float32)
    print(f'avg cosine similarity: {cosine.mean()}; in degree: {np.degrees(np.arccos(cosine.mean()))}')

def get_cosine_similarity(model, test_loader, num_classes, sample_mean, num_batches, layer_index, in_dist = False):
    '''
    only works for id
    '''
    model.eval()
    scores = torch.zeros(num_classes).cuda()
    counts = torch.zeros(num_classes).cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            out_features = model.intermediate_forward(data, layer_index)     # 64, 128
            if layer_index == 0:
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                out_features = torch.mean(out_features, 2)
        # compute cosine similarity score
        for i in range(len(target)):
            class_mean = sample_mean[target[i]] #sample_mean shape: # of cls, 128
            if scores[target[i]] == 0:
                scores[target[i]] = cos(out_features[i].view(1,-1), class_mean.view(1,-1))
                # score = torch.mm(out_features, batch_sample_mean.view(-1,1))
            else:
                scores[target[i]] += cos(out_features[i].view(1,-1), class_mean.view(1,-1)).squeeze()
            counts[target[i]] += 1  

    avg_scores = (scores.cpu()/counts.cpu()).mean()
    print(f'avg cosine similarity: {avg_scores.item()}; in degree: {np.degrees(np.arccos(avg_scores.item()))}')
        # cosine.extend(2-2*score.cpu().numpy())
    #DEBUG
    if in_dist:
        avg_dist = np.zeros(num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if j!= i: 
                    cos_score = cos(sample_mean[i].view(1,-1), sample_mean[j].view(1,-1)).squeeze()
                    degree = np.degrees(np.arccos(cos_score.item()))
                    avg_dist[i] += degree 
        avg_dist = avg_dist /(num_classes-1)
        print(f'Avg degree between mean vectors: {avg_dist}; ON avg: {avg_dist.mean()}')

    # #DEBUG
    #     all_prod = np.zeros((num_classes, num_classes))
    #     for i in range(num_classes):
    #         for j in range(num_classes):
    #             cos_score = cos(sample_mean[i].view(1,-1), sample_mean[j].view(1,-1)).squeeze()
    #             degree = np.degrees(np.arccos(cos_score.item()))
    #             all_prod[i,j] = degree
    #     print(f'degree among all mean vectors: {all_prod}')