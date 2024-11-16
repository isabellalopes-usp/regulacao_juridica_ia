from google.colab import drive
import os
import pandas as pd
import re

class DataPreprocessor:
    def __init__(self):
        #conectar ao drive
        drive.mount('/content/drive', force_remount=True)
        self.data_path = '/content/drive/MyDrive/TCC_FINAL/duplicates_identified'
        self.output_path = '/content/drive/MyDrive/TCC_FINAL/pre_processed_data'
        os.makedirs(self.output_path, exist_ok=True)

        #carregar planilha de termos e padroes regex
        self.termos, self.padroesRegex = self.load_terms_and_patterns()

    def load_terms_and_patterns(self):
        file_path = os.path.join(self.data_path, "termos_utilizados.xlsx")
        df_terms = pd.read_excel(file_path)
        termos = df_terms['termo'].tolist()
        padroesRegex = df_terms['padrao'].tolist()
        return termos, padroesRegex

    def preprocess_dataframe(self, df, pattern):
        #limpeza das colunas processo e idacordao
        df['processo'] = df['processo'].astype(str).str.replace(r'\D', '', regex=True)
        df['idacordao'] = df['idacordao'].astype(str).str.replace(r'\D', '', regex=True)

        for col in ['classe', 'assunto', 'relator', 'orgao_julgador', 'acordao', 'ementa','repeticoes']:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].str.lower()

        #ajustar a coluna acordao
        if 'acordao' in df.columns and df['acordao'].notna().any():
            df['acordao'] = df['acordao'].apply(self.clean_acordao)

        if 'ementa' in df.columns and df['ementa'].notna().any():
            df['ementa'] = df['ementa'].apply(self.clean_ementa)

        #buscar ocorrencias baseadas no termo e padrao
        df['ocorrencias'] = df['acordao'].apply(lambda x: self.buscar_ocorrencias(pattern, x))
        df['mencoes_cruzadas'] = [self.buscar_mencoes(df['processo'], df['acordao'], df['assunto'], i, df['processo'][i]) for i in range(len(df))]

        return df

    def clean_ementa(self, text):
        if pd.isna(text):
            return text
        #remover múltiplos espaços e quebras de linha consecutivas
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        unwanted_phrases = [r'\(TJSP.+\)']
        for phrase in unwanted_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)

        return text.strip()

    def clean_acordao(self, text):
        if pd.isna(text):
            return text

        #remover multiplos espaços e quebras de linha consecutivas
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        #remover trechos indesejados
        unwanted_phrases = [
            r'TRIBUNAL.+JUSTI.A.+PAULO',
            r'TRIBUNAL.+JUSTI.A',
            r'PODER\sJ.+O',
            r'AS.+ELETR.NICA',
            r'ac.rd.o\n',
            r'\s\srelator(a)\n',
            r'\s\s\srelator\n',
            r'apel.+\s.+\d',
            r'reg.+\d',
            r'voto.+\d',
            r'decl.+\d',
            r's.o.+\sde\s.+\sde\s20[0-2][0-5]',
            r'c.mara\n',
            r'c.mara:',
            r'.*c.mara.+justi.a\n',
            r'.*c.mara.+paulo\n',
            r'fl.*\d',
            r'\n..\n',
            r'\n.\n',
            r'\n\s*\n',
            r'\s\B'
        ]
        for phrase in unwanted_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)

        return text.strip()

    def formataProcesso(self,num):
        return re.sub(r"(\d{7})(\d{2})(\d{4})(\d{1})(\d{2})(\d{4})", r"\1-\2.\3.\4.\5.\6", num)

    def buscar_mencoes(self, processos, acordaos, assuntos, i, num):
        num = str(num)
        n_processo0 = num
        n_processo1 = self.formataProcesso(num)
        m = ""
        for l in range(len(acordaos)):
            if processos[l] != processos[i] and n_processo1 in acordaos[l]:
                acordaos[l] = re.sub(re.escape(n_processo1), n_processo0, acordaos[l], 1)
                t = str(self.buscar_ocorrencias(n_processo0, acordaos[l]))
                m += f"PROCESSO: {processos[l]}\nASSUNTO: {assuntos[l]}\nTRECHOS:\n{t}\n\n"

        return m

    def buscar_ocorrencias(self, pattern, texto):
        ocorrencias = re.findall(r'[.]?[\w\s\d\,\-\:\–\nº\(\)]*' + pattern + r'[\s\S]*?[.]', texto, re.IGNORECASE)
        return self.formatar_ocorrencias(ocorrencias)

    def formatar_ocorrencias(self, paragrafos):
        ocorrencias_formatadas = []
        for idx, paragrafo in enumerate(paragrafos, start=1):
            ocorrencia = f'ocorrencia {idx}: {paragrafo[1:]}'.replace("\n", " ")
            ocorrencias_formatadas.append(ocorrencia)
        return "\n\n".join(ocorrencias_formatadas) if ocorrencias_formatadas else ""

    def process_files(self):
        for file_name in os.listdir(self.data_path):
            if file_name.endswith('.csv'):
                #extrair o nome do arquivo e remover o prefixo
                base_name = file_name.replace('processed_com-acordaos-', '').replace('.csv', '').strip()

                #identificar o indice do termo correspondente
                if base_name in self.termos:
                    index = self.termos.index(base_name)
                    pattern = self.padroesRegex[index]
                else:
                    print(f"Termo '{base_name}' não encontrado na lista de termos.")
                    continue

                #ler o arquivo csv em um dataframe
                file_path = os.path.join(self.data_path, file_name)
                df = pd.read_csv(file_path)

                #preprocessar o dataframe e buscar ocorrencias
                df = self.preprocess_dataframe(df, pattern)

                #salvar o dataframe processado no novo caminho
                output_file = os.path.join(self.output_path, base_name.replace(' ', '_') + '.csv')
                df = df.drop_duplicates(subset=['idacordao'])
                df.to_csv(output_file, index=False)
                print(f"Arquivo salvo: {output_file}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process_files()
