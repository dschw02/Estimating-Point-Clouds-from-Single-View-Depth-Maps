input_layer 32 32 1
down_conv_2d 96 3 leaky_relu batch_norm
down_conv_2d 128 3 leaky_relu batch_norm
down_conv_2d 192 3 leaky_relu batch_norm
flatten
fully_connected 4096 leaky_relu batch_norm
fully_connected 2048 leaky_relu batch_norm
fully_connected 1024 leaky_relu batch_norm
fully_connected 2048 leaky_relu batch_norm
fully_connected 4096 leaky_relu batch_norm
expand 4 4
up_conv_2d 192 3 8 8 leaky_relu batch_norm
up_conv_2d 128 3 16 16 leaky_relu batch_norm
up_conv_2d 96 3 32 32 leaky_relu batch_norm
conv_2d 96 3 linear
end
