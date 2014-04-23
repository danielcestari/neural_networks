# criarei funcoes para a geracao de pontos que formem classes
#	nao linearmente separaveis para testar o perceptron


# Funcao que gera pontos situados entre duas curvas
##
# x		=> vetor com as posicoes em x que serao 
#			aplicadas as funcoes que limitam 
#			superior e inferiormente os pontos 
#			a serem gerados, como se fosse o
#			dominio da funcao
# fCima		=> funcao que limita a regiao por cima, 
#			a equacao de uma reta por ex.
# fBaixo	=> funcao que limita a regiao por baixo,
#			a equacao de uma reta por ex.
# pts_vertical	=> quantidade de pontos no eixo y a serem
#			gerados
# dist_uniforme	=> BOOLEAN que diz se a quantidade de pontos
#			no eixo y deve ser constante ou
#			ser aleatoria

gera_pontos <- function(x, fCima, fBaixo, pts_vertical=5, 
			dist_uniforme=TRUE){
	
	# DEBUG
	mute_cat <- function(...){}
	if(!exists('DEBUG')) DEBUG = FALSE;
	if(!DEBUG){
		original_cat <- cat
		cat <- mute_cat
	}
	
	pontos = c()
	for(i in x){
		# DEBUG
		cat('X atual:',i,'\n')
		cat('fCima:',fCima(i), ' fBaixo:', fBaixo(i),
			'\n\n')
		
		if(!dist_uniforme) 
			pts_vertical = floor(runif(1, 5, 20))
		y = seq(from=fBaixo(i), to=fCima(i), 
				length.out=pts_vertical)

		for(j in y)
			pontos = rbind(c(i, j), pontos)
	}

	
	# DEBUG
	if(!DEBUG) cat <- original_cat

	return (pontos)
}

#############################################################



# primeira classe sera formada por um triangulo
#	cada borda sera descrita por uma reta

# x=[0.5;1.2]

x_c1 = seq(from=0.5, to=1.2, by=0.05)

c1_baixo <- function(x){
	return (2*x+1)
}

c1_cima <- function(x){
	return (-0.5*x+4)
}

pts_c1 = gera_pontos(x_c1, c1_cima, c1_baixo, pts_vertical=5, 
			dist_uniforme=F)

#############################################################


# segunda classe sera formada por uma regiao delimitada
#	pela inteseccao de dois triagulo
# assim forma-se uma regiao nao convexa e combinado com 
#	a classe 1 nao eh linearmente separavel

x_c2 = seq(from=0.8, to=2, by=0.05)

c2_baixo1 <- function(x){
	return (1)
}

c2_cima1 <- function(x){
	return (3*x-0.6)
}

c2_baixo2 <- function(x){
	return (-x+5)
}

c2_cima2 <- function(x){
	return (5.5)
}

pts_c21 = gera_pontos(x_c2, c2_cima1, c2_baixo1, 
			pts_vertical=5, dist_uniforme=F)

pts_c22 = gera_pontos(x_c2, c2_cima2, c2_baixo2,
			pts_vertical=5, dist_uniforme=F)


#############################################################

# gerando conjunto de dados linearmente separavel

# primeira gera dataset da classe 1 (resposta para essa 
#	asse eh zero)
#	apenas precisa fazer o apend da ultima coluna


linear_c1 = cbind(pts_c1, 0)

linear_c2 = cbind(pts_c21, 1)


# duas abordagens na divisao dos datasets para treinamento
#	e para validacao, uma pego elementos sequencialemnte
#	de cada classe, primeiro X elementos da classe 1,
#	depois Y elementos classe 2, em ordem
# a outra abordagem seria juntar os 2 dos conjuntos, da
#	classe 1 e da classe 2 (linear_c1 e linear_c2), e 
#	selecionar os elementos desse bolo aleatoriamente

# a divisao no caso sequencial sera 70% para treinamento
#	e 30% para validacao

linear_dataset_seq=rbind(linear_c1[1:(nrow(linear_c1)*0.7),],
			linear_c2[1:(nrow(linear_c2))*0.7,])
linear_validate_seq=rbind(linear_c1[1:(nrow(linear_c1))*0.3,],
			linear_c2[1:(nrow(linear_c2))*0.3,])

# selecionando aleatoriamente
tmp = rbind(linear_c1, linear_c2)


# puxar os pontos da classe 1, "triangulo a esquerda" um pouco
# mais para a esquerda, 0.3 pontos para a esquerda
tmp[tmp[,3] == 0, 1] = tmp[tmp[,3] == 0, 1] -0.3


vector_size = nrow(tmp)
sample = floor( runif(vector_size*0.3, 1, vector_size+1) )

linear_validate_aleatorio = tmp[sample, ]
tmp = tmp[-sample,]

# se deixar assim o conjunto de treinamento fica sequencial,
# primeiro os da classe 1 em sequida da classe 2
# scrambling
vector_size = nrow(tmp)
sample = floor( runif(vector_size, 1, vector_size+1) )
linear_dataset_aleatorio = tmp[sample,]


dataset_linear = list()
dataset_linear$train = linear_dataset_aleatorio
dataset_linear$validate = linear_validate_aleatorio

#############################################################

# gerando conjunto de dados nao linearmente separavel

nlinear_c1 = linear_c1
nlinear_c2 = cbind(rbind(pts_c21, pts_c22), 1)

vector_size = floor(nrow(nlinear_c1)*0.3)
sample = floor( runif(vector_size, 1, vector_size+1) )
nlinear_validate = nlinear_c1[sample,]
nlinear_dataset = nlinear_c1[-sample,]

vector_size = floor(nrow(nlinear_c2)*0.3)
sample = floor( runif(vector_size, 1, vector_size+1) )
nlinear_validate = rbind(nlinear_validate, 
				nlinear_c2[sample,])
nlinear_dataset = rbind(nlinear_dataset, 
				nlinear_c2[-sample,])


vector_size = nrow(nlinear_validate)
sample = floor( runif(vector_size, 1, vector_size+1) )
nlinear_validate = nlinear_validate[sample,]


vector_size = nrow(nlinear_dataset)
sample = floor( runif(vector_size, 1, vector_size+1) )
nlinear_dataset = nlinear_dataset[sample,]



dataset_nlinear = list()
dataset_nlinear$train = nlinear_dataset
dataset_nlinear$validate = nlinear_validate

