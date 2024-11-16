import os
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

def identificar_repeticoes(data_path):
    #carregar arquivos
    arquivos_csv = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    #criar um dicionario
    processos_dict = {}

    for arquivo in arquivos_csv:
        file_path = os.path.join(data_path, arquivo)
        df = pd.read_csv(file_path)

        #coluna processo como string
        df['idacordao'] = df['idacordao'].astype(str)

        #armazenar os processos e as planilhas em que aparecem
        for processo in df['idacordao']:
            if processo in processos_dict:
                processos_dict[processo].append(arquivo)
            else:
                processos_dict[processo] = [arquivo]

    #adicionar a coluna repeticoes em cada arquivo
    for arquivo in arquivos_csv:
        file_path = os.path.join(data_path, arquivo)
        df = pd.read_csv(file_path)

        df['idacordao'] = df['idacordao'].astype(str)

        #adicionar a coluna repeticoes
        def encontrar_repeticoes(processo):
            if processo in processos_dict:
                #lista de planilhas onde o processo aparece, exceto a atual
                outras_planilhas = [p for p in processos_dict[processo] if p != arquivo]
                if outras_planilhas:
                    return ', '.join(outras_planilhas).replace("com-acordaos-","").replace(".csv","")
            return "nao repetido"

        df['repeticoes'] = df['idacordao'].apply(encontrar_repeticoes)

        #salvar
        output_path = os.path.join('/content/drive/MyDrive/TCC_FINAL/duplicates_identified', 'processed_' + arquivo)
        df.to_csv(output_path, index=False)
        print(f"Arquivo atualizado salvo em: {output_path}")

if __name__ == "__main__":
    data_path = '/content/drive/MyDrive/TCC_FINAL/scraped_data/raspagem_outubro'
    identificar_repeticoes(data_path)
