
# implementação da RBF em R

# TODO: fazer função para desenhar a fronteira de decisão,
#		ou seja, precisa fazer a inversa da função de base 
#		radial. G(x)*W=0 é a equação para a fronteira de decisão
# 		também é interessante desenhar os clusters com um círculo 
#		marcando o raio de 1 sigma

# TODO: verificar se compensa tentar fazer a rbf mais 
#		generalizada, com aprendizado supervisionado nas duas
#		etapas.

##
# Função que treina uma rede RBF
##
# dataset			=> Dataset a ser classificado
# labelCol			=> Primeira coluna com a classe. Caso 
#						tenha mais de uma saída, o código 
#						calcula a coluna das outras saídas
# saida				=> Número de neurônios na camada de saída
# centers_selection	=> Função que executa a seleção dos 
#						centróides
# k					=> Número de centróides (funções de base 
#						radial), dependendo do modo de 
#						seleção dos centróides não é utilizado
# sigma				=> Sigma usado na função de base radial
#						Caso o valor passado seja 0, então uma
#						seleção automática será feita
# threshold			=> Parâmetro para estabilização na escolha
#						dos centróides
# radial_function	=> Função de base radial
##
# Retorna os parâmetros da rede RBF treinados, centros, sigma 
# usado, matriz de pesos W
###

rbf_train <- function(dataset, labelCol, saida=1,
			centers_selection=random_selection, k=5, sigma=1, 
			threshold=0.01, radial_function=gaussian, 
			DEBUG=FALSE, ...){
	args = list(...)
	
    # apenas silencia a funcao cat caso a variavel DEBUG 
    #   seja FALSE
    mute_cat <- function(...){}
    if(!DEBUG) {
        original_cat <- cat
        original_print <- print
        cat <- mute_cat
        print <- mute_cat
    }

	
	# Usaremos a função ginv (Moore-Penrose generalized 
	# inverse of a matrix) desse pacote
	library(MASS)
	
	# Escolhe centróides
	# parte demorada da RBF
	centers = centers_selection(dataset[, 1:(labelCol-1)], k=k, 
					threshold=threshold)
	
	# caso sigma == 0, determina um valor para sigma 
	# automaticamente
	# Segue a estratégia de aprendizado especificado no livro 
	# do haykin seção 7.11 da primeira edição
	# Segundo o livro isso garante que a gaussiana não seja 
	# muito "peaked" ou muito "flat"
	if(sigma == 0){
		max_dist = 0
		for(c in 1:length(centers)){
			for(i in 1:nrow(dataset)){
				dist = sqrt(sum((centers[c] - dataset[i, 1:(labelCol-1)])^2))
				if(dist > max_dist){
					max_dist = dist
				}
			}
		}
		sigma = max_dist/sqrt(2*k)
	}

	cat('\nCentros:\n'); print(centers); cat('\n')

	# calcula matriz G
	# da equacao G*W=d
	G = c()
	for(i in 1:nrow(dataset)){
		g = c()
		for(j in 1:k){
			g = c(g, radial_function(
				dataset[i, 1:(labelCol-1)], centers[j, ], 
				sigma=sigma))
		}
		G = rbind(G, g)
	}
	# acrescenta coluna de "1", usado para o cálculo do bias
	G = cbind(G, 'bias'=1)
	
	cat('\nMatriz G:\n')
	print(G); cat('\n')

	# Encontra pesos da camada de saída
	W = ginv(G) %*% as.matrix(dataset[, labelCol:ncol(dataset)])
	
	# Retorna parâmetros da rede treinados
	ret = list()
	ret$centers = centers
	ret$sigma = sigma
	ret$W = W
	#ret$G = G
	#ret$ginv = ginv(G)
	#ret$d = dataset[, labelCol:ncol(dataset)]
	
	# antes de sair reestabelece a funcao cat original
    if(!DEBUG){
        cat <- original_cat
        print <- original_print
    }
	
	return(ret)
}

########################################################
########################################################
########################################################

##
# Dado os parâmetros da rede, e uma amostra classifica
##
# W					=> Matriz de pesos
# centers			=> Matrix com os centros, um por linha
# sigma 			=> Desvio padrão da função de base radial
# radial_function	=> Função de base radial que será usada
# sample			=> Exemplo que será classificado
##
# Retorna o vetor d da equação, g*W=d, que representa a saída
###
# TODO: permitir um sigma para cada centróide

rbf_test <- function(W, centers, sigma,
				radial_function=gaussian, sample, DEBUG=FALSE){
	
    # apenas silencia a funcao cat caso a variavel DEBUG 
    #   seja FALSE
    mute_cat <- function(...){}
    if(!DEBUG) {
        original_cat <- cat
        original_print <- print
        cat <- mute_cat
        print <- mute_cat
    }

	
	# calcula matriz G
	# da equacao g*W=d
	g = c()
	for(j in 1:nrow(centers)){
		g = c(g, radial_function(sample, centers[j, ], sigma=sigma))
	}
	# acrescenta coluna de "1", usado para o cálculo do bias
	g = c(g, 'bias'=1)
	
	#cat('\nW:\n'); print(W);cat('\n');
	#cat('\ng:\n'); print(g);cat('\n');
	
	# antes de sair reestabelece a funcao cat original
    if(!DEBUG){
        cat <- original_cat
        print <- original_print
    }
	
	
	return( g %*% W )
}



########################################################
########################################################
########################################################


##
# Faz validação da RBF para os parâmetros passados
##
# dataset 			=> Dataset usado para treinar e testar rede
# model				=> Modelo (parâmetros da rede treinada) de
#						mesmo que retornado pela função rbf_train
#						Se esse parâmetro for passado apenas 
#						checa o modelo para o dataset em questão
#						Ou seja, os parâmetros threshold e 
#						prop_training são ignorados
# labelCol          => Primeira coluna com a classe do exemplo
# centers_selection => Função que executa a seleção dos 
#                       centróides
# k                 => Número de centróides (funções de base 
#                       radial), dependendo do modo de 
#                       seleção dos centróides não é utilizado
# sigma             => Sigma usado na função de base radial
# threshold         => Parâmetro para estabilização na escolha
#                       dos centróides
# radial_function   => Função de base radial
# prop_training		=> Porcentagem do dataset usada para 
#						treinamento
##
# Retorna training_set, test_set, o modelo treinado (retorno 
# da função rbf_train), valor do erro (calculado pela diferença
# do esperado pela saída real) por iteracao e o erro acumulado 
# (total), e a porcentagem de acertos (calculada pegando 
# como resposta a saída de maior valor)
###

rbf_validate <- function(dataset, model=NULL, labelCol, 
						centers_selection=kmeans, 
						k=5, sigma=1, threshold=0.01, 
						radial_function=gaussian, 
						prop_training=0.7){
	
	# apenas separa os dados do dataset se um modelo não for 
	# passado para teste 
	if(is.null(model)){
		test_only = FALSE
		# separando training_set e test_set
		cat('\nSeparando dataset ...\n')
		dataset_size = nrow(dataset)
		training_set = sample(dataset_size, round(dataset_size*prop_training))
		test_set = dataset[-training_set, ]
		training_set = dataset[training_set, ]
		cat('Training_set: ', nrow(training_set), 'exemplos\n')
		cat('Test_set: ', nrow(test_set), 'exemplos\n')
	
		# treina rede RBF
		cat('\n\nTreinando a rede ...\n');
		saida = ncol(dataset) - labelCol
		model = rbf_train(dataset=training_set, labelCol=labelCol, 
					saida=saida, centers_selection=centers_selection, 
					k=k, radial_function=radial_function, 
					sigma=sigma, threshold=threshold)
		
		cat('Centróides encontrados\n')
		print(model$centers)
	} else{
		test_only = TRUE
		test_set = training_set = dataset
	}
	
	# testando rede treinada
	cat('\n\nTestando modelo treinado...\n')
	mean_squared_error = 0
	erro_iter = c()
	acertos = 0
	for(i in 1:nrow(test_set)){
		resp = rbf_test(W=model$W, centers=model$centers, 
			sigma=model$sigma, radial_function=radial_function,
			sample=test_set[i, 1:(labelCol-1)])
		
		# calcula erro iterativo e erro quadratico medio
		tmp = sum((test_set[i, labelCol:ncol(test_set)] - resp)^2)
		erro_iter = c(erro_iter, tmp)
		mean_squared_error = mean_squared_error + tmp 
		
		if(which.max(resp) == which.max(test_set[i, 
								labelCol:ncol(test_set)]))
			acertos = acertos + 1
	}
	acertos = acertos / nrow(test_set)
	mean_squared_error = mean_squared_error / nrow(test_set)
	cat('Porcentagem de acertos: ', acertos, '\n')
	cat('Erro quadrático médio: ', mean_squared_error, '\n')
	
	ret = list()
	# se for apenas para testar já tenho o modelo, test_set e training_set
	# em outras variáveis
	if(!test_only){
		ret$model = model
		ret$training_set = training_set
		ret$test_set = test_set
	}
	ret$acertos = acertos
	ret$mean_squared_error = mean_squared_error
	ret$erro_iter = erro_iter
	
	return(ret)
}

########################################################
############# Funções auxiliares #######################
########################################################

##
# Função de base radial gaussiana
##
# sample	=> Ponto a ser avaliado
# center	=> Centro da função
# sigma		=> Desvio padrão da função
##
# Retorna o valor da função no ponto "sample" para o centro
# em questão com desvio padrão "sigma"
###

gaussian <- function(sample, center, sigma=1){
	return(exp(-sum(sample-center)^2/sigma^2))
}

##
# Função que faz a seleção dos centróides aleatórios tirados 
# do dataset
##
# dataset 	=> Dataset
# k			=> Número de centróides a selecionar
##
# Retorna lista de centróides
###

random_selection <- function(dataset, k=1, ...){
	centers = dataset[sample(nrow(dataset), k), ]
	return(centers)
}


##
# Função de agrupamento kmeans
##
# dataset		=> Dataset
# K				=> Número de clusters que será gerado
# threshold		=> Valor da mudança dos centros que considera
#					estáveis
##
# Retorna lista com os centróides e os ids dos elementos em
# cada centróide
###

getClosestCentroid <- function(instance, centroids) {

    euclidean = rep(0, nrow(centroids))
    for (i in 1:nrow(centroids)) {
        euclidean[i] = sqrt(sum((centroids[i,] - instance)^2))
    }

    id = which.min(euclidean)

    return(id)
}

suppressWarnings(warning("kmeans"))
kmeans <- function(dataset, k, threshold = 0.1) {
    # criar k centroides aleatoriamente
    npoints = nrow(dataset)
    ids = sample(seq(1,npoints), size=k)

    # criando uma matrix que contem a posicao atual dos centroides
    centroids = as.data.frame(dataset[ids,], ncol=ncol(dataset))
    allIds = rep(0, npoints)

    div = threshold + 1
    while (div > threshold) {
        # para cada elemento no conjunto de dados
        # encontre qual eh o centroide mais proximo
        for (i in 1:npoints) {
            idCentroid = getClosestCentroid(dataset[i,], centroids)
            allIds[i] = idCentroid
        }

        #print(allIds)

        div = 0
        for (i in 1:nrow(centroids)) {
            ids = which(allIds == i)
            oldcentroid = centroids[i,]
            centroids[i,] = colMeans(as.data.frame(dataset[ids,], ncol=ncol(dataset)))

            if (is.nan(unlist((centroids[i,])))) {
                centroids[i,] = dataset[sample(seq(1,npoints), size=1),]
                div = div + threshold*1000;
            } else {
                div = div + sqrt(sum((centroids[i,] - oldcentroid)^2))
            }
        }

        #print(centroids)

        div = div / nrow(centroids)

        #cat("divergence: ", div, "\n")
    }

    # verificando qual o centroide que cada elemento do
    # dataset estah associado
    for (i in 1:npoints) {
        idCentroid = getClosestCentroid(dataset[i,], centroids)
        allIds[i] = idCentroid
    }
	
	return(centroids)
}



########################################################
########################################################
########################################################

##
# Dada uma configuração de rede faz um k-fold Cross-Validation
# O label do dataset precisa ser especificado por neurônio de saída.
# Ex: se tem 3 saídas, precisa ter três colunas como label, cada uma
# representando a saída do correspondente neurônio.
# Isso é assumido por essa função
##
# dataset
# k_fold
# 
##
# 
# 
###

# TODO: normaliza dados ??, Como a saída é um combinador linear acho que não precisa tanto

k_fold_cross_validation <- function(dataset, labelCol, k_fold=10, 
					saidas=1, k=5, sigma=1, centers_selection=kmeans, 
					threshold=0.01, radial_function=gaussian){
	
	cat('\nExecutado ', k_fold,'-fold Cross-Validation\n\n')
	
	
    # divide dataset em k sub-amostras
	cat('\nDividindo dataset\n')
	nexamples = nrow(dataset)
    subsets = list()
    subset_size = nexamples/k_fold
    total_col = ncol(dataset)
	for(i in 1:k_fold){
        index = sample(1:nrow(dataset), subset_size)
        subsets[[i]] = dataset[ index, ]

		dataset = dataset[ -index, ]
    }
	
	# itera k vezes montando um conjunto de treino e de teste
    experimentos = list()
    summary_test = c()
    summary_train = c()
    for(i in 1:k_fold){
        train_set = c()
		test_set = c()
        for(j in 1:k_fold){
            if(j != i){
                train_set = rbind(train_set, subsets[[j]])
            } else{
                test_set = subsets[[j]]
            }
        }

        experimentos[[i]] = list()
        experimentos[[i]]$train_set = train_set
		experimentos[[i]]$test_set = test_set
		
		
		# treina modelo com train_set
		model = rbf_train(dataset=train_set, labelCol=labelCol, 
					saida=saidas, k=k, centers_selection=centers_selection, 
					sigma=sigma, threshold=threshold, 
					radial_function=radial_function)
		experimentos[[i]]$model = model
		
		# valida modelo treinado com test_set
		results = rbf_validate(test_set, model=model, labelCol=labelCol,
                   #centers_selection=centers_selection,
                   #k=k, sigma=sigma, threshold=threshold,
                   radial_function=radial_function)
		
		# salvando resultados
		experimentos[[i]]$results = results
		
		
	}

	# TODO: computar max e min dos resultados, em relacao à porcentagem

	return(experimentos)
	
}



##
# Função que dado o dataset, os centros e o sigma, imprime 
# o dataset duas a duas variáveis desenhendo a duas fronteiras
# para cada centro. Uma com raio 1*sigma e outra com raio 
# 2*sigmas.
##
# dataset		=> 
# labelCol		=> 
# centers		=> 
# sigma			=> 
# W				=> 
# save_to_file	=> 
##
#
###

# TODO: desenhar a fronteira final, usando W

draw_bounders <- function(dataset, labelCol, centers, sigma, 
					W, save_to_file=F){
	
	# caso tenha mais de uma saída, nesse caso considero 
	# saídas binárias
	# crio um único array de saída, com a "classe" do elemento
	saidas = dataset[, labelCol]
	if(labelCol != ncol(dataset)){
		saidas = rep(0, times=nrow(dataset))
		for(saida in 1:(ncol(dataset) - labelCol +1)){
			saidas[ dataset[, (labelCol -1 + saida) ] == 1 ] = saida
		}
	}
	
	# itera entre todas as variáveis
	for(var1 in 1:(labelCol-2)){
		for(var2 in (var1+1):(labelCol-1)){
			dev.new()
			plot(dataset[, var1], dataset[, var2], col=saidas)
			#symbols(x=1, y=1, circles=c(5), add=T, inches=F)
			
			# se deve salvar em arquivo o grafico
			if(save_to_file){
				

				dev.off()
			}
		}
	}
}
