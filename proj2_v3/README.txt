Histogram Layer was implemented without any for loops.
Initially I tried to do what was written in the comments trying to find the index of where the max value occurred
But I kept running into errors and realized that it would be better using torch.max which can find the max along a dimension and return 0 for all other places. 
Created a binary occupancy tensor by filling it with 1 where the max value exists
And then multiplied the binary occupancy tensor with norm of image gradients to get the output
