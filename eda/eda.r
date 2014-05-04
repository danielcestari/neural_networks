# Script simples para realizacao de EDA
# Exploratory Data Analysis

####################
# dataset			=> 
# order_class		=> Flag se eh necessario ordernar as
#						classes
##
eda <- function(dataset, class_col=1, order_class=F, 
				draw_boxplot=F){
	
	if(order_class){
		dataset = dataset[order(dataset[,class_col]),]
	}

	result = list()
	
	spread = rbind(summary(dataset), 
				array(0, c(4,ncol(dataset))))

	print(spread)

	freq = list()
	for(col in 1:ncol(dataset)){
		x = dataset[,col]
		freq[[col]] = cbind(Freq=table(x), 
						Cumul=cumsum(table(x)), 
						relative=prop.table(table(x)))

		# adiciona informacoes de espalhamento
		spread[7,col] = paste('Median Abs. Dev.: ', 
							round(mad(x), 4))
		spread[8,col] = paste('SD: ', round(sd(x), 4))
		#spread[9,col] = paste('Range: ', (spread[6,col] - 
		#					spread[1,col]))
		#spread[10,col] = paste('IQR: ', (spread[5,col] - 
		#					spread[2,col]))
	}
	# TODO: fazer as frequencias combinadas.
	# para cada valor de uma variavel seleciona as outras
	# e entao traca a tabela de frequencia
	result[['frequency']] = freq
	
	result[['summary']] = spread
	
	if(draw_boxplot) boxplot(dataset)
	
	return (result)
}

