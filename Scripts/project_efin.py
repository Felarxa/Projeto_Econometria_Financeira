import yfinance as yf
from Scripts.functions import econometria_functions
from Scripts.functions import finance_functions
from datetime import date, timedelta

def main():
    start_date = (date.today() - timedelta(days=(5*365))).isoformat()
    end_date = date.today().isoformat()

    tickers = [
        'VALE3.SA',
        'ELET3.SA',
        'PETR4.SA',
        'COGN3.SA',
        'BBDC4.SA'
    ]

    output = {}
    try:
        # Baixar dados para o setor atual
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close'].dropna()

        # Verificar se há dados suficientes
        if data.empty or len(data) < 2:
            print("Dados insuficientes para cálculo")
            print(output)
            return
        
    except Exception as e:
        print(f"Erro na API - {str(e)}")


    for ticker in tickers:
        fin = finance_functions(ticker.split('.')[0])
        df_balanco, df_dre, df_fluxo_caixa = fin.get_financial_dataframes()
        df_balanco = fin.format_df_balanco(df_balanco)
        df_dre = fin.format_df_dre(df_dre)
        df_fluxo_caixa = fin.format_df_fluxo(df_fluxo_caixa)
        dfs_list = [df_balanco, df_dre, df_fluxo_caixa]
        dfs_list = [fin.format_dataframe(df) for df in dfs_list]
        df_balanco, df_dre, df_fluxo_caixa = dfs_list
        val_acao = fin.Val_Acao(df_balanco, df_dre, df_fluxo_caixa)
        output[ticker] = [ticker]
        output[ticker].append(f'Valor calculado da ação: {val_acao}')


    try:
        # Calcular pesos
        missing_tickers = [ticker for ticker in tickers if ticker not in data.columns]
        if missing_tickers:
            print(f"Os seguintes tickers não possuem dados - {', '.join(missing_tickers)}")
        weights = econometria_functions.minvariance(data, 0.005)

        # Formatar a saída
        for coluna, weight in zip(data.columns, weights.round(4)):
            output[coluna].append(f'Último valor de fechamento: {data[coluna].iloc[-1].round(4)}')
            output[coluna].append(f'Peso ótimo: {weight}')
            print(f'\n{output[coluna]}')

    except Exception as e:
        print(f"Erro no cálculo - {str(e)}")
    


if __name__ == '__main__':
    main()