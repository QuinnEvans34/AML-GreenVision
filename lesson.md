Input
layer 1
activation
layer 2
prediction
loss

gradient descent finds "how wrong was the prediction (loss)" then "how much did each weight contribute to that loss"
gradiant is just a slope

positive gradient = move weight down 
negative gradient = move weight up
large gradient = bigger update
smaller gradient = smaller update

adam w uses two memories to adjust the weight diffusion
what direction has the gradients been pointing?
what is the average size of each gradient?

mental model:
back propogation is the teacher, it grades the model and tells loss
adam says i will use that feedback, but i will smooth it and scale it based on past behavior
adam with l2 style says I will mix the weight shrinking penalty into the teachers fadback before processing it
adamw says I will keep the teachers feedback clean, process it with adam, and apply weight shrinkage


"A neural network learns in a loop that repeats thousands of times. Each loop has four phases: data goes in (forward pass), we measure how wrong we were (loss), we figure out which way to adjust each weight (backward pass), and then the optimizer applies that adjustment (update weights). Adam and AdamW both live in that fourth phase.
When the optimizer updates the weights, it also tries to keep weights from growing too large — because large weights are a sign of overfitting. That's called weight decay. AdamW and plain Adam both do weight decay, but plain Adam does it in a way that's unfair to weights that are being actively updated by gradients — those weights get less shrinkage, even though they're often the ones most at risk of overfitting. AdamW fixes the unfairness — every weight gets the same proportional shrinkage.
The demo shows just the weight-decay step, isolated. The big orange weight represents 'a connection that grew too large during training.' Watch what each optimizer does to it."


The optimizer is basically saying:
“Take the current weight, look at the gradient, scale that gradient by the learning rate, then subtract it from the weight.”


L2 is effectively the same thing as weight decay, you would apply this to the gradient and change weight from this product
