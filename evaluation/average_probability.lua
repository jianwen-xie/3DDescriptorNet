require './predict'

function get_classification_error(preds, category)
    assert(category <= 40)
    local err = 0
    local num_voxels = preds:size()[1]
    for tt = 1, num_voxels do
        local pred = preds:index(1, torch.LongTensor({tt}))
        local res, ind = torch.max(pred, 2)
        if not (ind[1][1] == category) then
            err = err + 1
        end
    end
    return err/num_voxels
end

function get_average_probability(preds, category)
   return torch.mean(preds:index(2, torch.LongTensor({category})))
end

class_table = {
    ['bathtub'] = 2,
    ['bed'] = 3,
    ['chair'] = 9,
    ['desk'] = 13,
    ['dresser'] = 15,
    ['monitor'] = 23,
    ['night_stand'] = 24,
    ['sofa'] = 31,
    ['table'] = 34,
    ['toilet'] = 36
}

function test_avg_probability()
    local current_data = loadMatFile(opt.syn_path, 'v')
    local preds = get_predictions(current_data)
    local class = class_table[opt.class]
    local classifcation_error = get_classification_error(preds, class)
    print(string.format("classifcation error: %f", classifcation_error))
    local average_probability = get_average_probability(preds, class)
    print(string.format("average probability: %f", average_probability))
end

test_avg_probability()