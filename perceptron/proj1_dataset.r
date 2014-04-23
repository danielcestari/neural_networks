# cada instancia sera a representacao de uma imagem de 5x5 
#	pixels
# cada pixel ou tera valor 1 ou 0

# o ultimo elemento de cada vetor representa se eh V ou V 
#	invertido, isso no dataset apenas

V1 = c(	1,0,0,0,1,
	0,1,0,1,0,
	0,1,0,1,0,
	0,1,0,1,0,
	0,0,1,0,0
	)

V2 = c(	1,0,0,0,1,
	1,0,0,0,1,
	0,1,0,1,0,
	0,1,0,1,0,
	0,0,1,0,0
	)

V3 = c(	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	0,1,0,1,0,
	0,0,1,0,0
	)

# Criando o dataset a partir desses tres vetores para o V
proj1_dataset = matrix(c(
				V1,1,
				V2,1,
				V3,1,
				rev(V1),0,
				rev(V2),0,
				rev(V3),0
				), 
				nrow=6, 
				ncol=(length(V1)+1), 
				byrow=TRUE)



# deixar esses dois para teste

V4 = c(	1,0,0,0,1,
	1,0,0,0,1,
	0,1,0,1,0,
	0,1,0,1,0,
	0,1,1,1,0
	)

V5 = c(	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	0,1,0,1,0,
	0,1,1,1,0
	)
# precisa adicionar a entrada referente ao Theta, na primeira 
#	posicao
proj1_validate = matrix(c(	
				V4,1,
				V5,1,
				rev(V4),0,
				rev(V5),0
				), 
				nrow=4, ncol=(length(V4)+1), 
				byrow=TRUE)

##############################################

# gera um arquivo com a saida desse, um .dat assim pode ser 
#	usado em outra linguagem tb, ',' (virgula) eh o 
#	separador dos campos, sendo que o primeiro jah eh do 
#	Theta


write(t(proj1_dataset), 'proj1_dataset.dat', 
	ncolumns=ncol(proj1_dataset), sep=',')

write(t(proj1_validate), 'proj1_validate.dat', 
	ncolumns=ncol(proj1_validate), sep=',')

