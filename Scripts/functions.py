import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from lxml import etree
import yfinance as yf

class econometria_functions():

    def construct_Q(returns):
        # converter para np
        returns_np = returns.to_numpy()

        # Número de ativos
        n_assets = returns_np.shape[1]

        # Matriz de covariância (equivalente a cov(return) em R)
        cov_matrix = np.cov(returns_np, rowvar=False)

        # Vetor de 1's (equivalente a rep(1, ncol(assets)))
        ones_row = np.ones((1, n_assets))

        # Médias das colunas (equivalente a colMeans(return))
        means_row = returns_np.mean(axis=0, keepdims=True)  # Funciona com arrays NumPy

        # Combinação vertical (equivalente a rbind)
        Q = np.vstack([cov_matrix, ones_row, means_row])

        # Transpor as últimas duas linhas da matriz
        last_two_rows_transposed = Q[-2:].T

        # Criar matriz de zeros 2x2
        zeros_block = np.zeros((2, 2))

        # Concatenar
        right_block = np.vstack([last_two_rows_transposed, zeros_block])

        # Concatenar com a matreiz Q original
        Q = np.hstack([Q, right_block])

        return Q

    def construct_b(returns, mu):
        # Número de ativos
        n_assets = returns.shape[1]

        # Vetor de zeros
        zeros_part = np.zeros(n_assets)

        # Concatenar
        b = np.concatenate([zeros_part, [1], [mu]])

        return b

    def minvariance(assets, mu):
        # Definindo número de assets
        n = len(assets.columns)

        # Calculate log returns
        returns = np.log(assets / assets.shift(1)).dropna()

        # Construct Q matrix
        Q = econometria_functions.construct_Q(returns)

        # Construct b vector
        b = econometria_functions.construct_b(assets, mu)

        # Solve the system of equations
        weights = np.linalg.solve(Q, b)

        # Return only the weights (excluding the Lagrange multipliers)
        return weights[:n]

class finance_functions():

    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_html(self):
        base_url = f"https://www.dadosdemercado.com.br/acoes/{self.ticker}"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar página de {self.ticker}: {e}")
            return None

    def extract_tables(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all(attrs={'id': ('balances', 'incomesYear', 'cashflows')})
        return tables

    def parse_table(self, table):
        rows = table.find_all('tr')
        table_data = []

        for row in rows:
            headers = row.find_all('th')
            if headers:
                cols = [cell.text.strip() for cell in headers]
            else:
                cols = [cell.text.strip() for cell in row.find_all('td')]
            table_data.append(cols)

        return pd.DataFrame(table_data)

    def get_financial_dataframes(self):
        html = self.fetch_html()
        if html is None:
            return []

        tables = self.extract_tables(html)
        return [self.parse_table(table) for table in tables]
    
    def convert_value(self, value):
        multp = {'t': 1e12, 'b': 1e9, 'm': 1e6, 'mil': 1e3}
        val_str = str(value).strip().lower()
        parts = val_str.split(' ')

        if len(parts) == 2:
            x, y = parts
            x = x.replace(',', '.')
            x = float(x) * multp.get(y, 1)
            return int(x)
        
        return value

    def format_dataframe(self, df):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df[df.columns[1:]] = df[df.columns[1:]].map(self.convert_value)
        return df
    
    def format_df_balanco(self, df):
        df = df.drop(df.columns[1], axis=1)
        df = df.drop(df.columns[6:], axis=1)
        df = df.drop(df.columns[2:5], axis=1)
        return df
    
    def format_df_dre(self, df):
        df = df.drop(df.columns[1], axis=1)
        df = df.drop(df.columns[3:], axis=1)
        return df
    
    def format_df_fluxo(self, df):
        df = df.drop(df.columns[3:], axis=1)
        return df

    def Val_Acao(self, df0_1, df1_1, df2_1):
        try:
            LL = df1_1[df1_1.columns[1]][10]
            Dep_A = df2_1[df2_1.columns[1]][1]
            CapEx = sum(df0_1[df0_1.columns[1]][[9,10]])-sum(df0_1[df0_1.columns[2]][[9,10]])
            Cap_G = (df0_1[df0_1.columns[1]][1] - df0_1[df0_1.columns[1]][12]) - (df0_1[df0_1.columns[2]][1] - df0_1[df0_1.columns[2]][12])
            Div_Liq_E = ((df0_1[df0_1.columns[1]][14] + df0_1[df0_1.columns[1]][16])-(df0_1[df0_1.columns[2]][14] + df0_1[df0_1.columns[2]][16]))-(df0_1[df0_1.columns[1]][2] - df0_1[df0_1.columns[2]][2])
            FCFE = LL + Dep_A - CapEx - Cap_G + Div_Liq_E
            ################################################
            EBIT = df1_1[df1_1.columns[1]][4]
            Tx_Imp = abs(df1_1[df1_1.columns[1]][7]) / df1_1[df1_1.columns[1]][6]
            FCFF = EBIT * (1 - Tx_Imp) + Dep_A - CapEx - Cap_G
            ################################################
            info = self.get_info()
            beta = info.get('beta')
            selic = self.get_selic()
            Mkt_Price = 0.05
            ke = (beta * Mkt_Price) + (selic/100)
            Div_Liq = (df0_1[df0_1.columns[1]][14] + df0_1[df0_1.columns[1]][16]) - df0_1[df0_1.columns[1]][2]
            P_Liq = df0_1[df0_1.columns[1]][17]
            W_Div = Div_Liq / (Div_Liq + P_Liq)
            W_P_Liq = P_Liq / (Div_Liq + P_Liq)
            WACC = (selic/100) * W_Div * (1 - Tx_Imp) + W_P_Liq * ke
            ################################################
            YG = self.sect_tx(info)
            LT_G = YG/2
            ################################################
            P_FCFF = [FCFF*(1+YG)]
            for i in range(0,4):
                P_FCFF.append(P_FCFF[i]*(1+YG))
            present_value = []
            for i in range(0,5):
                present_value.append(P_FCFF[i]/(1+WACC)**(i+1))
            ################################################
            Val_Terminal = P_FCFF[4]*(1+LT_G)/(WACC-LT_G)
            ValP_Terminal = Val_Terminal/(1+WACC)**5
            Val_Presente_Total = sum(present_value)+ValP_Terminal
            Eq_Val = Val_Presente_Total-Div_Liq
            n_acoes = self.get_n_acoes()
            ################################################
            Val_Acao = Eq_Val/n_acoes
            return Val_Acao.round(2)
        except Exception as e:
            print(f"Erro no cálculo de valuation da {self.ticker}- {str(e)}")
        
        return 'null'
    
    def get_info(self):
        variations = [
            self.ticker,
            f"{self.ticker}.SA"
        ]
        
        for variation in variations:
            try:
                dat = yf.Ticker(variation)
                info = dat.info
                beta = info.get('beta')
                if beta is not None:
                    print(f"Info from {self.ticker} successfully retrieved")
                    return info
            except Exception:
                continue

        return None

    def get_selic(self):
        url = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados/ultimos/1?formato=json'
        resp = requests.get(url)
        selic = float(resp.json()[0].get('valor')) + 0.1
        return selic
    
    def sect_tx(self, info):
        setores = {
            'Basic Materials': 0.031,
            'Communication Services': 0.081,
            'Consumer Cyclical': 0.047,
            'Consumer Defensive': 0.035,
            'Energy': 0.034,
            'Financial Services': 0.11,
            'Healthcare': 0.053,
            'Industrials': 0.04,
            'Real Estate': 0.103,
            'Technology': 0.109,
            'Utilities': 0.08
        }
        sector = info.get('sector')
        tx = setores.get(sector, 0.01)
        return tx
    
    def get_n_acoes(self):
        html = self.fetch_html()
        dom = etree.HTML(str(html))
        n_acoes = dom.xpath('/html/body/div[3]/div[1]/div[7]/span[2]')[0].text
        n_acoes = int(''.join(n_acoes.split('.')))
        return n_acoes