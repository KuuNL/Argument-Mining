import json
import itertools
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    file1 = open('gold/', 'r')
    truth = json.load(file1)['AIF']
    file2 = open('test/baseline_ilo/', 'r')
    preds = json.load(file2)

    proposition_dict = {}
    proposition_list = []
    locution_dict = {}
    locution_list = []
    true_illocution_list = []
    pred_illocution_list = []

    # Get the list of proposition and locution nodes
    for node in truth['nodes']:
        if node['type'] == "I":
            proposition_list.append(node['nodeID'])
            proposition_dict[node['nodeID']] = node['text']
        elif node['type'] == "L":
            locution_list.append(node['nodeID'])
            locution_dict[node['nodeID']] = node['text']

    proploc_list = proposition_list + locution_list

    # Check truth illocutions
    for node in truth['nodes']:
        if node['type'] == "YA":
            illocution_id = node['nodeID']
            illocution_type = node['text']

            for edge in truth['edges']:
                if edge['fromID'] == illocution_id:
                    target_id = edge['toID']
                    for edge in truth['edges']:
                        if edge['toID'] == illocution_id:
                            source_id = edge['fromID']
                            if source_id in proploc_list and target_id in proploc_list:
                                true_illocution_list.append([source_id, target_id, illocution_type])
                    break

    # Check predicted illocutions
    for node in preds['nodes']:
        if node['type'] == "YA":
            illocution_id = node['nodeID']
            illocution_type = node['text']

            for edge in preds['edges']:
                if edge['fromID'] == illocution_id:
                    target_id = edge['toID']
                    for edge in preds['edges']:
                        if edge['toID'] == illocution_id:
                            source_id = edge['fromID']
                            if source_id in proploc_list and target_id in proploc_list:
                                pred_illocution_list.append([source_id, target_id, illocution_type])
                    break

    # print(true_illocution_list)
    # print(pred_illocution_list)

    p_c = itertools.product(locution_list, proposition_list)
    proploc_combinations = []
    for p in p_c:
        proploc_combinations.append([p[0], p[1]])

    y_true = []
    y_pred = []
    for comb in proploc_combinations:
        added_true = False
        added_pred = False

        # Prepare Y true
        for illocution in true_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_true.append(illocution[2])
                added_true = True
                break

        if not added_true:
            y_true.append('None')

        # Prepare Y pred
        for illocution in pred_illocution_list:
            if illocution[0] == comb[0] and illocution[1] == comb[1]:
                y_pred.append(illocution[2])
                added_pred = True
                break

        if not added_pred:
            y_pred.append('None')

    print(y_true)
    print(y_pred)

    focused_true = []
    focused_pred = []
    for i in range(len(y_true)):
        if y_true[i] != 'None':
            focused_true.append(y_true[i])
            focused_pred.append(y_pred[i])

    print(focused_true)
    print(focused_pred)

    print("General", precision_recall_fscore_support(y_true, y_pred, average='macro'))
    print("Focused", precision_recall_fscore_support(focused_true, focused_pred, average='macro'))

