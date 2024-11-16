install.packages("remotes")
install.packages("readxl")
remotes::install_github("jjesusfilho/tjsp", force=TRUE)
remotes::install_github("leeper/rio", force=TRUE)

library(readxl)
p <- read_excel("termos_utilizados.xlsx")
ter <- p$termo

verificaDiretorio = function(t) {
  if (dir.exists(t)) {
    unlink(t) # Se o diretório já existe, exclui
  }
  dir.create(t) # Cria um novo diretório
}

criaDataFrame = function(tabela,vetor_acordaos,textos){
  df <- data.frame(
    classe = tabela$classe,
    assunto = tabela$assunto,
    relator = tabela$relator,
    comarca = tabela$comarca,
    orgao_julgador = tabela$orgao_julgador,
    data_julgamento = tabela$data_julgamento,
    data_publicacao = tabela$data_publicacao,
    processo = tabela$processo,
    ementa = tabela$ementa,
    idacordao = vetor_acordaos,
    acordao = textos
  )
  return(df)
}

leituraAcordaos = function(t,i,j){
    nome <- paste(t,"/","cdacordao_", i, ".pdf", sep = "") #armazena nome do arquivo baixado, com diretorio

    while(file.info(nome)$size == 0){
      file.remove(nome)
      tjsp_baixar_acordaos_cjsg(i, diretorio = t) #baixa arquivo referente ao codigo do acordao
    }

    leitura <- paste(" ", (ler_acordaos(arquivos = nome,remover_assinatura = TRUE,combinar = TRUE))$julgado, sep="") #armazena apenas a leitura do texto do acordao

  return(leitura)
}

buscaAcordaos = function(tabela,t){
  vetor_acordaos_char <- tabela$cdacordao
  vetor_acordaos <- as.integer(vetor_acordaos_char)
  textos <- c()

  j<-1
  for(i in vetor_acordaos){ #percorre os resultados, a fim de baixar os respectivos acordaos
      tjsp_baixar_acordaos_cjsg(i, diretorio = t) #baixa arquivo referente ao codigo do acordao
        tryCatch({
          textos <- c(textos,leituraAcordaos(t,i,j))
        }, error = function(e) {
          textos <- c(textos,"erro na leitura do acordao")
        })
       #le arquivo e atualiza resultados
      j<-j+1
  }
  df <- criaDataFrame(tabela,vetor_acordaos,textos)
  return(df)
}

processa_termo = function(t) {
    
    nome_final <- paste("com-acordaos-", t, ".csv", sep = "")
    if (!file.exists(nome_final)) {
        verificaDiretorio(t)
        busca <- paste('"', t, '"', sep = "") #constroi string de busca
        tjsp_baixar_cjsg(livre=busca, diretorio=t)
        tabela <- tjsp_ler_cjsg(diretorio = t)
        vetor_processos <- tabela$processo

        if (length(vetor_processos) > 0) {
            df <- buscaAcordaos(tabela, t)
            export(df, nome_final)
        } else {
            unlink(t, recursive = TRUE)
        }
    }
}

library(tjsp)
library("rio")

for(t in ter) {
  autenticar(login="", password="") #credenciais ocultas
  tryCatch({
    processa_termo(t)
  }, error = function(e) {
    cat(paste("termo nao funcionou:", t))
  })
}
