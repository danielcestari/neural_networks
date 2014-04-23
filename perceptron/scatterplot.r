
# inicialmente apenas para perceptron

# TODO: deixar generico de modo e ser usado por outros metodos
#	depende da maneira que os outros metodos separam os dados

# executa funcao que gera scatterplot com dados de treinamento e de validação
# os dados de treinamento estarao em vermelho e os de validacao em azul

# cada classe tera um simbolo diferente, os dois primeiros sao bola e quadrado
# até o limite de 4 classes, pch=15,16,17,18


# dataset	=> tipo lista, cada elemento sendo um conjunto de pontos de uma determinada classe


scatterplot <- function(train_dataset, validate_dataset){
	
	sink('plot.pl')
	
	# TODO: pegar o range das variaveis
	cat('set xr [-1:+1]\n')
	cat('set yr [-1:+1]\n')

	# TODO: pegar o nome das dimensoes
	cat('set xlabel "A"\n')
	cat('set ylabel "B"\n')
	cat('set zlabel "C"\n')

	# funcao do hiperplano, nesse caso plano mesmo
	cat('hiperplano(x,y) = ', W[1], ' + ', W[2], '*x + ', W[3], '*y\n')

	cat('spl '
	
	"-" using 1:2:3')
	
	# precisa jah no comando spl especificar qtas entradas tera
	# entao precisa saber qtas classes serao inseridas para fazer o loop
	
	cat('"-" using 1:2:3')
	
	cat('')
	
	sink()

	
}

 


