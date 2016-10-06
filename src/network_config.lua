require 'torch'
require 'nn'
require 'Rectifier'

function cnn_config()
    local args={}
    args.input_dims={1,12,1}
    args.n_units        = {6, 16}
    args.filter_size    = {8, 2}
    args.filter_size_v = {1, 1}
    args.filter_stride  = {1, 1}
--    args.n_units        = {32, 64, 64}
--    args.filter_size    = {2, 2, 2}
--    args.filter_stride  = {1, 1, 1}
--    args.filter_size    = {8, 4, 3}
--    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {72}
    args.nl             = nn.Rectifier
     
    return create_network(args)
 
end


function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack( args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

     net:add(convLayer(1, args.n_units[1],
                        args.filter_size_v[1],args.filter_size[1], 
                        args.filter_stride[1], args.filter_stride[1],1)) 
--    net:add(nn.ReLU())
       net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size_v[i+1],args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
--        net:add(nn.ReLU())
         net:add(args.nl())
    end

    local nel = net:forward(torch.zeros(1,unpack( args.input_dims))):nElement()
    
    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
--    net:add(nn.ReLU())
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
--        net:add(nn.ReLU())
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, 3))

        print(net)
        print('Convolutional layers flattened output size:', nel)
    return net
end
