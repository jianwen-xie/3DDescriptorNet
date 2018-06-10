require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
require 'hdf5'
require 'torch'

assert(pcall(function () mat = require('fb.mattorch') end) or pcall(function() mat = require('matio') end), 'no mat IO interface available')

cmd = torch.CmdLine()
cmd:option('-syn_path', '', 'Path to synthesized results.')
cmd:option('-model_path', './3dnin_fc/3dnin_fc.net', '3D classification model path.')
cmd:option('-batch_size', 20, 'Batch size for computing average prediction results.')
cmd:option('-class', '', 'Object category.')

opt = cmd:parse(arg)

cutorch.setDevice(1)

model = torch.load(opt.model_path):cuda()
model:add(cudnn.SoftMax():cuda())

function get_predictions(voxels)
    -- assert(voxels:nDimension() == 4)
    assert(torch.max(voxels) <= 1.0)
    assert(torch.min(voxels) >= 0.0)

    model:evaluate()
    voxels = voxels:csub(voxels:min()) / (voxels:max() - voxels:min())
    local preds = torch.LongTensor():cuda()
    local num_voxels = voxels:size(1)
    print(voxels:size())
    local indices = torch.randperm(num_voxels):long():split(opt.batch_size)
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        
        local inputs = voxels:index(1, v)
        local inputs = inputs:reshape(inputs:size(1),1,30,30,30)
        local output = model:forward(inputs:cuda())
        preds = preds:cat(output, 1)
    end
    print(string.format('num_voxels: %d num_classes: %d', preds:size(1), preds:size(2)))
    return preds
end

function loadHdf5File(file_name, data_name)
    local current_file = hdf5.open(file_name,'r')
    local current_data = current_file:read(data_name):all():float()
    current_file:close()
    return current_data
end

function loadMatFile(file_name, data_name)
    local input = mat.load(file_name)[data_name]:float()
    print(string.format("Data Loaded, size: %d", input:size()[1]))
    return input
end