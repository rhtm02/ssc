import tensorflow as tf

def training(model,train_dataset,val_dataset,epoch=100,save_name='seq2sqe.h5'):
    BEST_MODEL = model
    optimizer = tf.keras.optimizers.RMSprop()
    print("*****************학습시작*****************")
    for epoch in range(epoch):
        loss_val_checker = 1000
        for step, (X_EN_BATCH,X_DE_BATCH, Y_BATCH) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits,_,_ = model([X_EN_BATCH,X_DE_BATCH])
                #print(X_EN_BATCH.shape,X_DE_BATCH.shape,Y_BATCH.shape,logits.shape)
                loss_val = tf.keras.losses.MSE(Y_BATCH, logits)
                loss_score = tf.math.reduce_mean(loss_val)

            grads = tape.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_mse = tf.keras.metrics.MSE(Y_BATCH, logits)
            if (step % 200) == 0:
                print('while one batch loss mean : ', loss_score.numpy())

            if(loss_val_checker > loss_score.numpy()):
                tf.saved_model.save(model, '../result/')
                #model.save('../result/' + save_name)
                print('model saved, loss:', loss_score.numpy())
                loss_val_checker = loss_score.numpy()
                BEST_MODEL = model
        # validation 평가
        for val_en_x_batch,val_de_x_batch, val_y_batch in val_dataset:
            #print(val_en_x_batch.shape,val_de_x_batch.shape, val_y_batch.shape)
            val_en_output, state_h, state_c = model.layers[0](val_en_x_batch)
            val_logits= []
            for _ in range(12):
                temp_logits,h,c = model.layers[1](val_de_x_batch, initial_state=[state_h, state_c])
                val_logits.append(temp_logits)
                val_de_x_batch = temp_logits
                state_h = h
                state_c = c
        val_mse = tf.keras.metrics.MSE(val_y_batch, val_logits)
        print("Validation mse :",tf.math.reduce_mean(val_mse).numpy)
        print("Train mse :",tf.math.reduce_mean(train_mse).numpy())
        #reset

    return BEST_MODEL

