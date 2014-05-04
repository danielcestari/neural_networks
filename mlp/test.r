mlp_validate(dataset=XOR_dataset, W=mlp_train(dataset=XOR_dataset, dimension=c(2,1), epochs=100, entrada_theta=-1, act_param=c(1,6)), dimension=c(2,1), entrada_theta=-1, act_param=c(1,6))




mlp_validate(dataset=XOR_dataset, W=array(c(1.5,1,1, 0.5,1,1, 0.5,-2,1, 0,0,0), c(3,2,2)), dimension=c(2,1), entrada_theta=-1, act_param=c(1,3))


W=mlp_train(dataset=XOR_dataset, dimension=c(2,1), w_init_zero=F, epochs=200, DEBUG=T, act_param=c(1,20), eta_para=0.001); mlp_validate(dataset=XOR_dataset, W=W, dimension=c(2,1), entrada_theta=-1, act_param=c(1,20), DEBUG=F)



epochs=1; eta=1; xor=XOR_dataset; W=mlp_train(dataset=xor, dimension=c(2,1), w_init_zero=array(c(2,2,1, 2,1,2, 2,1,1, 0,0,0, 0,1,0, 0,0,1), c(3,2,3)), epochs=epochs, DEBUG=T, act_param=c(1,200), eta_para=eta, epsilon=0); mlp_validate(dataset=xor, W=W, dimension=c(2,1), act_param=c(1,200), DEBUG=F)




# pesos retornados pelo MLP.java do mello 
mlp_validate(dataset=XOR_dataset, W=array(c(5.433330111439093,-3.691361292433329,-3.6897527790753966, 1.8708037899446393,-5.196210412834658,-5.189754821417071, -3.0100902173497395,6.706419155717727,-7.209678866891632, 0,1,0, 0,0,1 ), c(3,2,3)), dimension=c(2,1), entrada_theta=1, act_param=c(1,4))


epochs=20000; eta=0.1; dimension=c(4,2,3); alpha=0.5; epsilon=0.1; W=mlp_train(dataset, dimension=dimension, w_init_zero=F, epochs=epochs, DEBUG=F, act_param=c(1,1), eta_para=eta, alpha_momentum=alpha, epsilon=epsilon); mlp_validate(dataset, W=W, dimension=dimension, act_param=c(1,1), DEBUG=F)
