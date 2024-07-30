import math
import torch
import Constants as C


def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k, dcg_k, map, ap = 0, 0, 0, 0

    # Please check consistency with your baselines.
    # All baselines and this method followed this setting to compute the ideal DCG through a truncation.
    # (See Line. 94 (https://github.com/Coder-Yu/SELFRec/blob/main/util/evaluation.py))
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        map = ap / count

    return float(count / k), float(count / len(test_matrix)), map, float(dcg_k / idcg_k)


def vaild(prediction, label__,label, top_n, pre, rec, map_, ndcg):
    top_ = torch.topk(prediction, top_n, -1, sorted=True)[1]

    # new_label = 

    # print("top size : ",top_.size())
    
    # print("ground list shape : ",label__.size())
    # print("\nexamples : ",label[0])
    # print("\n",label[1])

    # i=0
    # print("\nchecking in pre rec top\n")
    for top, l in zip(top_, label__):
    # for top, l in zip(top_, label):
        # if len(l)==0:
        #     continue
        try:
            # print("org ",label[i])
            # print("try ", l)
            # i+=1
            l = l[l != 0] - 1
        except Exception as e:
            l = l[l != 0]
        recom_list, ground_list = top.cpu().numpy(), l.cpu().numpy()
        if len(ground_list) == 0:
            continue
        
        # print("ground list shape : ",label.size())

        # map2, mrr, ndcg2 = metric.map_mrr_ndcg(recom_list, ground_list)


################ MUST UNCOMMENT ######################################        
        pre2, rec2, map2, ndcg2 = precision_recall_ndcg_at_k(top_n, recom_list, ground_list)
        # pre2, rec2, map2, ndcg2 = precision_recall_ndcg_at_k(top_n, recom_list, label__)
        pre.append(pre2), rec.append(rec2), map_.append(map2), ndcg.append(ndcg2)

    # print("Recomended list : ",recom_list)

    # print("Recommended list shape : ",prediction.size())
    # print()
    # print()

    # # print("ground list : ",ground_list)

    # print("ground list shape : ",label.size())
    # print()
    # print()


def pre_rec_top(pre, rec, map_, ndcg, prediction, label__, label, event_type):

    # print("--------------------Inside pre rec top--------------------")
    # filter out the visited POI
    target_ = torch.ones(event_type.size()[0], C.POI_NUMBER, device='cuda:0', dtype=torch.double)
    # target_ = torch.ones(event_type.size()[0], C.USER_NUMBER, device='cuda:0', dtype=torch.double)
    # print()
    # print("event_type : ")
    # print("event type (in pre rec top) :",event_type)

    # for i in range(5):
    #     print(f"event type (in pre rec top) {i}:",event_type[i])
    # # print()
    # print("\nInside pre rec top\n")
    for i, e in enumerate(event_type):
        # print(i,e)
        e = e[e!=0]-1
        target_[i][e] = 0

    # print("prediction size : ",prediction.size())
    # print("target size : ",target_.size())
    
    #_________________________________________________________________
    prediction = prediction * target_
    prediction_ = torch.transpose(prediction, 0, 1)
    #___________________________________________________________________

    # print()
    # print("prediction shape : ",prediction.size())
    # print()

    # print()
    # print("label shape : ",label.size())
    # print()

    

    # print()
    # print("new prediction shape : ",transposed_tensor.size())
    # print()    

    # for i, topN in enumerate([1, 5, 10, 20]):
    # for i, topN in enumerate([1, 5]):


    #_________________________________________________
    for i, topN in enumerate([1]):
        vaild(prediction_, label__, label, topN, pre[i], rec[i], map_[i], ndcg[i])
