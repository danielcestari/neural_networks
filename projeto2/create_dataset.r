# dado o arquivo 'file_name' gera um dataset de treinamento e
# de teste
###############
# file_name			=> nome do arquivo que sera lido o dataset
#						
# class_col			=> numero da coluna da classe
# classes			=> matriz dando a posicao de inicio e fim
#						de cada classe, por linha
# percent_train		=> porcentagem do dataset usadao para 
#						treino
# sep				=> separador usado pela funcao read.table
##
create_dataset <- function(file_name, class_col, classes, 
							percent_train=0.7, sep=''){
	tmp = read.table(file_name, sep=sep)
	tmp = cbind(tmp[,-class_col], tmp[,class_col])
	
	# separa as classes do dataset
	class = list()
	train = c()
	test = c()
	for(i in 1:nrow(classes)){
		class_size = classes[i,2] - classes[i,1] +1
		class[[i]] = list()
		train_index = sample(1:class_size, 
								class_size*percent_train)
		test_index = (1:class_size)[-train_index]
		train =rbind(train, tmp[classes[i,1]+train_index -1,])
		test = rbind(test, tmp[classes[i,1]+test_index -1,])
	}
	
	dataset = list()
	dataset[['train']] = train
	dataset[['test']] = test
	
	return (dataset);
}

