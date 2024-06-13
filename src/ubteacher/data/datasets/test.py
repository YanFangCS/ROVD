with open('rovd_train_labeled_file_list.txt', 'r') as f:
    labeled_l = f.readlines()
    labeled_l = [t.strip() for t in labeled_l]

with open('rovd_train_file_list.txt', 'r') as f:
    all_l = f.readlines()
    all_l = [t.strip() for t in all_l]
    
with open('rovd_train_unlabeled_file_list.txt', 'r') as f:
    un_l = f.readlines()
    un_l = [t.strip() for t in un_l]

print(set(labeled_l) - set(all_l))
# print(len(labeled_l), len(un_l), len(all_l))