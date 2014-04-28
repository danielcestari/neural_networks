# Implementacao de um MLP
# 
# A modelagem sera matricial, entao nao ha necessidade de
# divisao em classes


# Funcoes heaviside e eta_cte copiadas do arquivo perceptron.r


##################################################
# Funcao de ativacao linear
linear <- function(value, ...){
	return (value)
}

##################################################

# Funcao de threshold Heaviside
# retorna 1 para valores maiores que o threshold, default 0.5
# a amplitude dessa funcao eh um (intervalo de saida de 0 a 1)
heaviside <- function(value, threshold=0.5){
	
	# aceita value como um vetor
	if(is.vector(value)){
		result = value
		result[value >= threshold] = 1
		result[value < threshold] = 0
		return (result)
	}
	
	
	if(value >= threshold) return (1);
	return (0);
}

###################################################
# Funcao de ativacao logistica

logistic <- function(value, param=c(1,1), ...){
	return (param[1]/(1+exp(-param[2]*value)));
}

###################################################
# Funcao derivada da ativacao logistica

logistic_deriv <- function(value, param=c(1,1), ...){
	return (param[2]*value*(1-value))
}

###################################################
# Funcao que retorna o eta como funcao de n (iteracoes)

# n		=> representa a example atual
# param		=> o parametro de controle da funcao
eta_cte <- function(n, param=0.1){
	return (param)
}

###########################################################
# Funcao que executa a fase Forward
#
# Deve retornar um vetor coluna representando o Y, matriz
#	com as saidas de todas as camadas
mlp_forward <- function(example, W, Y_inicial, 
						entrada_theta=-1, act_func=logistic,
						act_param=c(1,1), labelCol,n_camadas, 
						n_neuronio_max){

	
	# apenas silencia a funcao cat caso a variavel DEBUG 
	#	seja FALSE
	if(!exists('DEBUG')) DEBUG = FALSE
	mute_cat <- function(...){}
	if(!DEBUG) {
		original_cat <- cat
		original_print <- print
		cat <- mute_cat
		print <- mute_cat
	}

	# debug
	cat('\nn_camadas:', n_camadas, ' labelCol:', labelCol,
		' n_neuronio_max:', n_neuronio_max)
	cat('\nexample: ', example, '\nY_inicial:\n')
	print(Y_inicial)
	cat('\nW:\n'); print(W)

	# a primeira coluna eh a entrada, ou 
	#	Y0 
	Y = Y_inicial
	
	# debug
	cat('\nlabelCol: ',labelCol,'\n')
	cat('example:',example, '\n')
	
	cat('\n\t\tFASE FORWARD\n')
	
	
	Y[1:labelCol,1] = cbind( 
		c(entrada_theta, example[1:(labelCol-1)]))
	
	# calcula as saidas por camada 
	# (coluna), sendo a primeira camada
	# a propria entrada
	for(k in 1:n_camadas){


		# debug
		cat('\n\nk:', k,'\nt(W[,,k]):\n'); print(t(W[,,k])); cat('\nY[,k]:\n'); print(cbind(Y[,k]))


		Y[2:(n_neuronio_max+1),k+1] = 
			act_func(t(W[,,k]) %*% cbind(Y[,k]), act_param)
		# como os pesos dos neuronios
		# nao pertencentes a camada
		# em questao sao zero, nao ha
		# problema aqui.
		# mas atencao para manter
		# tais pesos zerados na fase
		# backwards
	}
	
	
	# antes de sair reestabelece a funcao cat original
	if(!DEBUG){
		cat <- original_cat
		print <- original_print
	}

	return (Y)
}

###########################################################
# Funcao de treino da MLP
##
# dataset	=> dataset usado para treinar a rede, a ultima
#			coluna ter a resposta do exemplo em 
#			questao
# dimension	=> vetor em que cada elemento representa o 
#			numero de neuronios na respectiva
#			camada, ex: c(2,2,1) dois neuronios 
#			na primeira camada escondida, dois 
#			neuronios na segunda camada escondida 
#			e uma saida
# epochs	=> numero de epocas que a rede sera treinada,
#			uma epoca corresponde a uma passada 
#			pelos exemplos do dataset
# eta_func	=> funcao que retorna o valor do parametro eta
#			no aprendizado, default eta eh 
#			constante permite eta variar com o
#			numero de iteracoes
# eta_param	=> parametro passado a funcao eta
# act_func	=> funcao de ativacao usadao, default funcao
#			heaviside
# act_param	=> parametro passado a funcao de ativacao
# w_init_zero	=> TRUE para inicializar o vetor de pesos (W)
#			com zero, FALSE para inicializa-los
#			aleatoriamente, esse parametro tb
#			permite passar um vetor de pesos ja
#			definido (usado na animacao), precisa
#			ser vetor coluna
# entrada_theta	=> valor que sera usado como entrada do 
#			parametro theta

# TODO: permitir mudar eta e parametros para cada camada
#	e depois para cada neuronio
# TODO: permitir mudar funcao de ativacao para cada camada
#	e depois para cada meuronio

# TODO: melhorar inicializacao dos pesos seguindo boas
#	praticas, questao do intervalo de inicializacao

mlp_train <- function(dataset, dimension, epochs=10, 
			eta_func=eta_cte, eta_para=0.1,
			act_func=logistic, act_param=c(1,1), 
			act_derivative=logistic_deriv, 
			w_init_zero=F, entrada_theta=-1,
			alpha_momentum=0, epsilon=0.01){
	
	
	# apenas silencia a funcao cat caso a variavel DEBUG 
	#	seja FALSE
	if(!exists('DEBUG')) DEBUG = FALSE
	mute_cat <- function(...){}
	if(!DEBUG) {
		original_cat <- cat
		original_print <- print
		cat <- mute_cat
		print <- mute_cat
	}

	# adiciona uma coluna no dataset referente ao
	# parametro Theta, entrada tem valor -1
	#dataset = cbind(rep(-1, nrow(dataset)), dataset)
	
	# labelCol guarda a posicao da primeira coluna de
	# resposta, lembrando normalmente tem mais de uma
	# Cada coluna de resposta representa a saida de um
	# neuronio da camada de saida
	labelCol = ncol(dataset) - 
		dimension[length(dimension)] +1
	# adiciona o numero de entradas no vetor dimension
	dimension = c(labelCol-1, dimension)
	
	
	# inicializar matriz tridimensional W
	# cada dimensao (slice) representa uma camada
	
	# como os calculos serao matriciais, eh preciso pegar
	# a maior valor de dimension para limitar a matriz
	
	# lembrando que cada elemento de dimension representa
	# o numero de neuronio na respesctiva camada
	
	# o +1 no fim da linha eh para adicionar uma linha
	# de entrada para o Theta para todos os neuronios
	n_neuronio_max = max(dimension[-1])
	n_entradas = max(dimension) +1
	n_camadas = length(dimension)
	
	# inicia vetor de pesos com zero, randomly ou por uma
	# matrix passada como parametro
	total_pesos = n_entradas * n_neuronio_max * n_camadas
	if(length(w_init_zero) > 1){
		W = w_init_zero
	} else{
			if(w_init_zero){
				pesos = rep(0, total_pesos)
			} else{
				pesos = runif(total_pesos, 0, 1)
			}
	}

	# TODO: aqui que posso colocar informacao a priori
	
	W = array(pesos, c(n_entradas, n_neuronio_max, 
			n_camadas))
	
	# a ultima camada eh uma matrix identidade, usada no
	# calculo do delta_atual
	W[,, n_camadas] = 0
	# a primeira linha eh relativa ao Theta, entao nao 
	# considero
	W[2:(n_neuronio_max+1), 1:n_neuronio_max, n_camadas] = 
					diag(n_neuronio_max)

	# criando matriz tridimensional delta W
	delta_W = delta_W_momentum = array(0, c(n_entradas, 
		n_neuronio_max,	n_camadas))

	# debug
	cat('\nInicializacao de W:\n')
	print(W)
	cat('\n\n')

	# PROVAVELMENTE ESSE TRECHO SERA USADO NOVAMENTE
	# percorre W zerando, repeitando o numero de neuronios
	# por camada
	for(k in 1:(n_camadas-1)){
		
		# adiciona um na entrada para representar o 
		# Theta
		entradas = dimension[k] +1
		neuronios = dimension[k+1]
		if(entradas != n_entradas)
			W[(entradas+1):n_entradas,,k] =	c(0)
		if(neuronios != n_neuronio_max)
			W[,(neuronios+1):n_neuronio_max,k] = c(0)

		# debug
		cat('\n camada:', k)
		cat('\nentradas: ', entradas, '\nneuronios:',
			neuronios, '\nn_neuronios_max:', 
			n_neuronio_max, '\nn_entradas:', 
			n_entradas, '\n')
	}

	
	#debug 
	cat('\n\ndataset: \n'); print(dataset)
	
	# cria matriz Y, com as saidas dos 
	#	neuronios para coluna 
	#	representa uma camada
	# e a primeira linha eh a entrada do
	#	Theta
	# como n_entradas eh maior que n_neuronio_max
	#	nrow deve ser n_entradas
	Y_inicial = matrix(0, nrow=n_entradas, 
			ncol=length(dimension))
	Y_inicial[1,] = entrada_theta

	# roda o numero de epocas solicitado
	iteracao = 1
	for(e in 1:epochs){
		# erro usado na condicao de parada
		# erro quadratico medio para uma epoca completa
		average_error = 0
		
		# debug
		cat('\n#############################')
		cat('\n\t\tEpoca: ',e,'\n')

		# iteracoes, uma para cada exemplo do dataset
		for(example in 1:nrow(dataset)){
			
			####################
			# fase forward
			####################
			
			
			Y = mlp_forward(dataset[example,], W, Y_inicial, 
						entrada_theta, act_func, act_param, 
						labelCol, n_camadas-1, n_neuronio_max)
			
			# debug
			cat('\n\nY:\n')
			print(Y)
			
			cat('\nFIM FASE FORWARD\n')
			cat('#################\n')
			cat('#################\n')
			
			##########################
			# fase backward
			##########################

			# debug
			cat('\n\n\t\tFase BACKWARD\n')

			# primeiro calculo o erro e atribuo
			# ao delta_anterior
			erro = cbind(
				dataset[example, labelCol:ncol(dataset)] - 
				Y[2+(1:tail(dimension, n=1)), n_camadas] )
			delta_anterior = matrix(0, nrow=n_neuronio_max,
				ncol=1)
			delta_anterior[1:length(erro)] = erro

			# atuliza erro quadratico medio
			average_error = average_error + t(erro) %*% erro

			# calcula a derivada de Phi
			Phi_linha = act_derivative(
				Y[2:nrow(Y), 2:ncol(Y)])
	
			# debug
			cat('Phi_linha\n'); print(Phi_linha)

			# calcula os vetores delta e tb ja calcula o 
			# delta_W para cada camada
			# tem que iterar de tras para frente
			for(j in (n_camadas-1):1){
				
				# debug
				cat('\n##############################\n')
				cat('\n\t\tCamada: ',j)
				cat('\ndelta_anterior:'); print(delta_anterior)
				cat('\nW[,,j+1]:\n'); print(W[,,j+1])

				# calcula o delta atual, da camada j, sendo 
				# analizada
				# ATENCAO: nao uso a primeira 
				# linha pq ela eh a entrada
				# em relacao a Theta, e nao
				# se calcula delta para o Theta
				delta_atual = (W[-1,,j+1] %*% delta_anterior)*
					Phi_linha[,j]
				
				# debug
				cat('\ndelta_atual:\n'); print(delta_atual)

				# salva a atualizacao dos pesos dessa camada
				# CHECAR SE NAO HA ATUALIZACAO DE PESOS ONDE NAO DEVE, EX NOS NEURONIOS QUE NAO EXISTEM
				for(i in 1:length(delta_atual)){
					delta_W[,i,j] = eta_func(iteracao) * 
						delta_atual[i] * Y[,j]
				}

				# faz delta_anterior igual delta_atual para a
				# proxima example
				delta_anterior = matrix(0, 
					nrow=n_neuronio_max, ncol=1)
				delta_anterior[1:length(delta_atual)] = 
					delta_atual

				# debug
				cat('\ndelta_W[,,j+1]:\n'); print(delta_W[,,j+1])
			}
			

			# fim dos calculos para o exemplo em questao
			
			# Calculo de delta_W com termo momentum
			# o valor default de alpha_momentum eh zero
			delta_W = alpha_momentum * delta_W_momentum + 
						delta_W
			delta_W_momentum = delta_W

			# atuliza os pesos W para a proxima iteracao
			W = W + delta_W

			# debug
			cat('\n\t\tAtualizacao dos pesos\n')
			cat('delta_W:\n'); print(delta_W)
			cat('\n\nW:\n'); print(W)
			cat('\n\t\t\t\tErro na iteracao:', erro)
		}
		
		# debug
		cat('\n\n\t\tFim da epoca: ', e)
		cat('\n\t\tErro acumulado na epoca: ',average_error/(2*nrow(dataset)))
				cat('\n\nprint:');print(average_error)	
		
		# implementar condicao de parada
		average_error = average_error / (2*nrow(dataset))
		if(average_error < epsilon){
			break
		}

		iteracao = iteracao +1
	}


	
	# antes de sair reestabelece a funcao cat e print original
	if(!DEBUG){
		cat <- original_cat
		print <- original_print
	}

	return (W[,,-n_camadas])
}


###########################################################
# Funcao que valida os pesos do MLP

mlp_validate <- function(dataset, W, dimension,
			entrada_theta=-1, act_func=logistic, 
			act_param=c(1,1), erro_threshold=0.3){
	
	resp = list()
	resp$average_error = 0
	resp$execucao = c()

	n_colunas = ncol(dataset)
	n_entradas = n_colunas - tail(dimension, n=1)


	labelCol = n_entradas +1
	dimension = c(n_entradas, dimension)
	n_neuronio_max = max(dimension[-1])
	n_camadas = length(dimension)
	
	Y_inicial = matrix(0, nrow=n_entradas+1, 
			ncol=length(dimension))
	Y_inicial[1,] = entrada_theta
	

	# erro quadratico medio total
	average_error = 0
	for(i in 1:nrow(dataset)){
		example = dataset[i, ]
		D = dataset[i, labelCol:n_colunas]
		Y = mlp_forward(example, W, Y_inicial, 
				entrada_theta, act_func, act_param, 
				labelCol, n_camadas-1, n_neuronio_max)
		
		# debug
		cat('\n\n\t\t\t\tY calculado:\n');print(Y)
		cat('\n\t\t\tn_camadas:',n_camadas,'\n\n')
		cat('\t\t\t\tdimension:', dimension,'\n')
		cat('\t\t\t 2+(1:tail(dimension, n=1)', 2+(1:tail(dimension, n=1)))
		cat('\n\n\n\t\t\t\t\ttail:', tail(dimension, n=1), '\n')

		# calcula o vetor erro para o dado exemplo
		erro = D - Y[1+(1:tail(dimension, n=1)), n_camadas]
		valor_erro = t(erro) %*% erro
		
		STR = 'ACERTOU'
		if(valor_erro > erro_threshold) STR = 'ERROU'

		resp$execucao = rbind(resp$execucao, 
							c(Y[1+(1:tail(dimension, n=1)), 
							n_camadas], D,valor_erro, STR))
		
		average_error = average_error + valor_erro
	}
	average_error = average_error / (2*nrow(dataset))
	resp$average_error = average_error

	return (resp)
}

###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
