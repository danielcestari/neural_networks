# arquivo que implementa o perceptron de uma camada

# o dataset sera uma matriz com cada linha sendo
#	um vetor de valores do exemplo de entrada,
#	com o ultimo elemento a classe que o exemplo
# 	da linha representa


#	para entrada de imagens, concatenar os pixels 
#	em um unico vetor


# a funcao de treino retorna o vetor de pesos 


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

# Funcao que retorna o eta como funcao de n (iteracoes)

# n		=> representa a iteracao atual
# param		=> o parametro de controle da funcao
eta_cte <- function(n, param=0.1){
	return (param)
}

###################################################

# Funcao que calcula a saida do perceptron

perceptron <- function(W, X, act_func=heaviside, 
			act_param=0.5){


	#cat('PERCEPTRON')
	#print(W)
	#print(X)

	return ( act_func((X %*% W), act_param) )
}


###################################################

# Funcao que treina uma rede perceptron de uma camada
##
# dataset	=> dataset usado para treinar a rede, a ultima
#			coluna ter a resposta do exemplo em 
#			questao
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

perceptron_train <- function(dataset, epochs=10, 
				eta_func=eta_cte, 
				eta_param=0.1, 
				act_func=heaviside, 
				act_param=0.5, 
				w_init_zero=T){
	
	# apenas silencia a funcao cat caso a variavel DEBUG 
	#	seja FALSE
	if(!exists('DEBUG')) DEBUG = FALSE
	mute_cat <- function(...){}
	if(!DEBUG) {
		original_cat <- cat
		cat <- mute_cat
	}
	
	# adiciona uma coluna no dataset, equivalente a 
	#	entrada do parametro Theta
	dataset = cbind(rep(1, nrow(dataset)), dataset)
	
	# Inicializa os pesos
	# as opcoes permitem inicializa-los com zero ou 
	#	aleatoriamente com valores entre 0 e 1
	labelCol = ncol(dataset)
	if(length(w_init_zero) > 1){
		# nesse caso recebe o vetor de pesos
		W = w_init_zero
	} else{
		# inicia vetor de pesos com zero ou randomly
		if(w_init_zero){
			W = cbind(rep(0, labelCol-1))
		} else{
			W = cbind(runif(labelCol-1, 0, 1))
		}
	}

	cat('\t\tPERCEPTRON: ', length(w_init_zero) > 1,'\n')
	print(w_init_zero)
	print(W)

	number_examples = nrow(dataset)
	iteration = 0
	for(e in 1:epochs){
		# DEBUG
		cat('Epoca: ',e,'\n')
		
		# checa se houve atualizacao dos pesos
		update = FALSE
		
		for(i in 1:number_examples){
			# DEBUG
			cat('\tExemplo: ',i,'\n')
			
			iteration = iteration +1
			
			Xn = dataset[i, 1:(labelCol-1)]
			Yn = perceptron(W, Xn, act_func,
				act_param)
			Dn = dataset[i, labelCol]
			
			# DEBUG
			cat('\tXn:', Xn, '\n\tW:',W, 
				'\n\tYn:', Yn, '\n')
			
			if(Yn == Dn) next
			
			# checa se houve atualizacao dos pesos
			update = TRUE
			
			delta_W = eta_func(iteration, 
				eta_param)*(Dn - Yn)*Xn
			W = W + delta_W
			
			# DEBUG
			cat('\n\tdelta_W:', delta_W, '\n\tW:',
				W, '\n\n')
		}
		
		# se nao houve atualizacao dos pesos para
		#	execucao
		if(!update) break
	}
	
	# msg de finalizacao
	cat('\n\nTreinamento finalizado\n', iteration, 
		'Iteracoes realizadas\nPesos encontrados:\n',
		W, '\n')
	
	# antes de sair reestabelece a funcao cat original
	if(!DEBUG) cat <- original_cat
	
	return (W)
}


###################################################

# dataset	=> dataset de treinamento do perceptron
# act_func 	=> funcao de ativacao
# act_param	=> threshold a ser passado para a funcao de 
#			ativacao
# w_init_zero 	=> TRUE se inicializa o vetor de pesos com 
#			zero caso contrario sera inicializado 
#			aleatoriamente funcao que valida o 
#			perceptron treinado dado um dataset de
#			validacao

perceptron_validate <- function(dataset, W, 
				act_func=heaviside,
				act_param=0.5){
	
	porcent_acertos = 0
	resp = list()
	resp$acertos = 0
	resp$execucao = c()
	
	# adiciona coluna no dataset referente ao Theta
	dataset = cbind(rep(1, nrow(dataset)), dataset)
	
	labelCol = ncol(dataset)
	for(i in 1:(nrow(dataset))){
		X = dataset[i, 1:(labelCol-1)]
		Y = perceptron(W, X, act_func, act_param)
		D = dataset[i, labelCol]

		str = 'ERROU'
		if(Y == D){
			str = 'ACERTOU'
			resp$acertos = resp$acertos +1
		}
		
		resp$execucao = rbind(resp$execucao, c(Y, D,
					str))
		
	}
	resp$acertos = resp$acertos / nrow(dataset)

	# colocando nome nas colunas para facilitar 
	#	visualizacao
	dimnames(resp$execucao) <- list(1:nrow(resp$execucao),
			c('Atual', 'Desejada', 'String'))
	
	return (resp)
}

####################################################

# Funcao que desenha a reta que representa os pesos 
#	retorna um conjunto de pontos
# 
# inicio	=> inicio da reta no eixo x
# fim		=> fim da reta no eixo x
# W		=> vetor de pesos, a convencao eh o primeiro
#			elemento eh Theta (b), o segundo x,
#			e o terceiro y

draw_line <- function(inicio, fim, W){
	x = seq(from=inicio, to=fim, length.out=100)
	a = -W[2]/W[3]
	b = -W[1]/W[3]
	pts = c()

	for(i in 1:100){
		y = a*x[i]+b
		pts = rbind(c(x[i], y), pts)
	}

	return (pts)
}

####################################################

# Funcao que anima o treinamento do perceptron
# a cada exemplo de treinamento e de validacao a reta 
# relativa aos pesos eh desenhada juntamento com os exemplos
# O plot eh limpo a cada epoch
##
# time_step	=> tempo de espera entre iteracoes, em 
#			segundos
# dataset	=> lista com dois campos, train e validate
#			cada qual usado para treinar e validar
# epochs	=> 
# eta_func	=> 
# eta_param	=> 
# act_func	=> 
# act_param	=> 

perceptron_animate <- function(	time_step=1, 
				dataset, 
				epochs=10, 
				eta_func=eta_cte, 
				eta_param=0.1, 
				act_func=heaviside, 
				act_param=0.5,
				w_init_zero=T){
	
	# time_step, epochs e w_init_zero  nao serao usados 
	# nas funcoes perceptron_train e perceptron_validate
	
	#treino = cbind(rep(1, nrow(dataset$train)), 
	#		dataset$train)
	#validacao = cbind(rep(1, nrow(dataset$validate)), 
	#		dataset$validate)
	treino = dataset$train[-(1:12),]
	validacao = dataset$validate
	
	labelCol = ncol(treino)
	# inicia vetor de pesos com zero ou randomly
	if(w_init_zero){
		W = cbind(rep(0.000001, labelCol))
	} else{
		W = cbind(runif(labelCol, 0, 1))
	}
	
	# pega o range de x e y, usado no calculo da reta de W
	xR = -Inf	# x right
	xL = Inf	# x left
	yU = -Inf	# y upper
	yL = Inf	# y lower
	for(i in 1:nrow(treino)){
		if(xR < treino[i,1]) xR = treino[i,1]
		if(xL > treino[i,1]) xL = treino[i,1]
	
		if(yU < treino[i,2]) yU = treino[i,2]
		if(yL > treino[i,2]) yL = treino[i,2]
	}
	for(i in 1:nrow(validacao)){
		if(xR < validacao[i,1]) xR = validacao[i,1]
		if(xL > validacao[i,1]) xL = validacao[i,1]
	
		if(yU < validacao[i,2]) yU = validacao[i,2]
		if(yL > validacao[i,2]) yL = validacao[i,2]
	}
	xR = xR + 2
	xL = xL - 2
	yU = yU + 2
	yL = yL - 2
	# define o range do grafico
	plot(seq(xL,xR, length.out=5), seq(yL,yU, 
			length.out=5), type='n')
	
	for(e in 1:epochs){
		
		cat('Epoca: ',e,'\n')
		
		# se nao houver alteracao dos pesos por uma
		# passagem sobre o dataset, terminar execucao
		updates = 0

		for(i in 1:nrow(treino)){
			
			cat('\tExemplo numero: ', i)
			cat('\tEpoca: ', e, '\n\n')

			exemplo = matrix(treino[i,], nrow=1, 
					ncol=labelCol)

			cat('EXEMPLO:',exemplo,'\n')
			cat('W:', W,'\n')
			
			W_anterior = W

			W = perceptron_train(exemplo, 1,
				eta_func=eta_func,
				eta_param=eta_param,
				act_func=act_func,
				act_param=act_param,
				w_init_zero=W)
				
			# verifica se houve mudanca em W
			updates = updates + 
				sum(W_anterior != W)

			cat('Novo W:',W,'\n')

			# apos treinar para um exemplo desenho
			# o exemplo e a reta relativo ao W
			color = 'red'
			if(exemplo[3] == 1) color = 'blue'
			points(exemplo[1], exemplo[2], 
				pch='*', col=color)

			points(draw_line(inicio=xL, fim=xR, 
				W), pch='*', col='green')

			# TODO: mudar a cor e o estilo dos
			# graficos de acordo com a classe


			# entre um exemplo e outro espera
			Sys.sleep(time_step)
		}

		# antes de comecar a proxima epoca, deixar
		# o grafico um pouco mais ressaltando o ultimo
		# vetor de pesos encontrado
		points(draw_line(inicio=xL, fim=xR, W), 
				pch='*', col='black')
		Sys.sleep(5)


		# apos o fim de um epoch limpo o grafico
		#frame()
		if(updates != 0 && e != epochs){
			# define o range do grafico
			plot(seq(xL,xR, length.out=5), 
					seq(yL,yU, 
					length.out=5), 
					type='n')
		}
		cat('\nFim da execucao da Epoca: ',e,'\n')
		

		# termina execucao caso nao houve mudancas
		if(updates == 0) break
	}
	

	cat('\nFim da animacao\n\n')
}




####################################################
