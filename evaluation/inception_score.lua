require './predict'

splits = 10

function get_inception_score(preds)
    local scores = torch.DoubleTensor(splits):fill(0)
    local num_voxels = preds:size(1)

    for i = 1, splits do
        local part = preds:sub(torch.floor((i-1) * num_voxels / splits + 1), torch.floor(i * num_voxels / splits))
        local kl = torch.cmul(part, torch.csub(torch.log(part), torch.log(torch.mean(part, 1):expand(part:size()))))
        kl = torch.mean(torch.sum(kl, 2))
        scores[i] = torch.exp(kl)
    end
    return torch.mean(scores), torch.std(scores)
end

function test_inception_score()
    local current_data = loadMatFile(opt.syn_path, 'v')
    local preds = get_predictions(current_data)
    local mean, std = get_inception_score(preds)
    print(string.format("mean: %f, std: %f", mean, std))
end

test_inception_score()