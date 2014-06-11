
# Implementação da PCA utilizando redes neurais artificiais

##
# Função que calculo as componentes principais de um dado
# dataset
##
# dataset		=> Dataset do qual as componentes serão 
#					extraídas
# eta 			=> Taxa de "aprendizado" para a convergência 
#					do algoritmo
# epochs		=> Número de épocas que o algoritmo executa
# epsilon 		=> Variação da matriz de pesos que define uma 
#					condição de parada
# components	=> Número de componentes calculadas
##
# Retorna as componentes principais do dataset em questão.
# Executa até o número de épocas especificado ou até a 
# variação dos pesos ser menor que epsilon.
###

pca_components <- function(dataset, eta=0.1, epochs=100, epsilon=0.05, 
					components=2){
	
	# inicializa matriz de pesos
	# as colunas representas os pesos do neurônio
	# ex: coluna 1 => pesos do neurônio 1
	dataset = as.matrix(dataset)
	W_size = ncol(dataset) * components
	W = matrix(runif(min=-1, max=1, n=W_size), ncol=components,
			nrow=ncol(dataset))
	delta_W = matrix(c(0), ncol=components, nrow=ncol(dataset))
	
	W_old = delta_W
	for(e in 1:epochs){
		
		# computa Y (saída), saída será uma matriz com cada
		# coluna representando a saída de um neurônio, e
		# cada linha a saída para o i-ésimo exemplo
		Y = dataset %*% W
		
		cat('\nY\n'); print(Y)
		cat('\nW\n'); print(W)


		# calculando o ajuste dos pesos
		for(n in 1:nrow(dataset)){
			x = dataset[n, ]
			x_ = x - (W %*% Y[n, ])
			
			for(j in 1:components){
				delta_W[, j] = eta*Y[n, j]*x_ - 
							eta*Y[n, j]*Y[n, j] * W[, j]
			}
			
			cat('\nx\n'); print(x)
			cat('\nx_\n'); print(x_)
			cat('\ndelta\n'); print(delta_W)
			
			# ajustando pesos
			W = W + delta_W

			cat('\nW\n'); print(W)

			#readline()
		}

		W = W / sqrt(sum( W^2 ))
		
		# condição de parada, variação de W é menor que 
		# a especificada epsilon, verificada ao final de 
		# cada época
		
		cat('\nEpoca: ', e)
		cat('\nAaa\n', norm(W-W_old), '\n')
		cat('\nW \n'); print(W)
		cat('\nW_old: \n'); print(W_old)

		if(sum((W - W_old)^2) <= epsilon){
			break
		}
		
		W_old = W
	}
	
	
	return(W)
}




"
bb = prcomp(iris[, 1:4])

dt1 = c()
for(i in 1:nrow(iris)){
	
	dt1 = rbind(dt1, sum(bb$rotation[, 1] * iris[i, 1:4]))
}


dt2 = c()
for(i in 1:nrow(iris)){
	dt2 = rbind(dt2, sum(bb$rotation[, 2] * iris[i, 1:4]))
}
dt2 = cbind(dt1, dt2)


dt3 = c()
for(i in 1:nrow(iris)){
    dt3 = rbind(dt3, c(sum(bb$rotation[, 3] * iris[i, 1:4]), sum(bb$rotation[, 4] * iris[i, 1:4])))
}
dt3 = cbind(dt2, dt3)




cc = prcomp(wine[, 2:14])

dt1_ = c()
for(i in 1:nrow(wine)){
    dt1_ = rbind(dt1_, sum(cc$rotation[, 1] * wine[i, 2:14]))
}


dt2_ = c()
for(i in 1:nrow(wine)){
    dt2 _= rbind(dt2_, sum(cc$rotation[, 2] * wine[i, 2:14]))
}
dt2_ = cbind(dt1_, dt2_)


dt3_ = c()
for(i in 1:nrow(wine)){
    dt3_ = rbind(dt3_, c(sum(cc$rotation[, 3] * wine[i, 2:14]), sum(cc$rotation[, 4] * wine[i, 2:14])))
}   
dt3_ = cbind(dt2_, dt3_)



"
