from tfautoencoder.autoencoder import getModel, getScaler

input_dimensionality = 200

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, input_dimensionality])

num_steps = 1000
learning_rate = 0.11
display_step = 10

# Construct model
scaler_op = getScaler(X)
mdl = getModel(scaler_op, input_dimensionality, nh=[50], lr=learning_rate)
encoder_op = mdl.encoder_op
decoder_op = mdl.decoder_op

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = scaler_op

# Define loss and optimizer, minimize the squared error
loss = mdl.loss
optimizer = mdl.optimizer

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4

# Start Training
# Start a new TF session                # Start a new TF session

with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

	# Training
	for i in range(1, num_steps + 1):
		# Prepare Data
		# Run optimization op (backprop) and cost op (to get loss value)
		for subs in range(10):
			sess.run(optimizer, feed_dict={X: avgvec})

		# Display logs per step
		if i % display_step == 0 or i == 1:
			l = sess.run(loss, feed_dict={X: avgvec})
			print('Step %i: Minibatch Loss: %f' % (i, l))
			if l < 0.015:
				break

	# Produce embedded vectors
	embedded = sess.run(encoder_op, feed_dict={X: avgvec})

