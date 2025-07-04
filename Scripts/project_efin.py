from functions import econometria_functions, finance_functions, general_functions
from datetime import date, timedelta

def main():
    ########################################
    # Altere a variável 'years_back' para  #
    # pegar dados de 'x' anos no passado   #
    ########################################
    years_back = 0
    base_date = date.today() - timedelta(days=(years_back*365))
    start_date = (base_date - timedelta(days=(1*365))).isoformat()
    end_date = base_date.isoformat()
    selic_date = (base_date + timedelta(days=1)).strftime("%d/%m/%Y") if base_date.weekday() == 6 else base_date.strftime("%d/%m/%Y")
    # Pega todos os tickers do link 'https://www.dadosdemercado.com.br/acoes' com mais de 1 milhão de Negócios.
    tickers = general_functions.get_tickers()

    output = {}
    html_dict = {}

    # Acessar as paginas de todos os tickers e garantir que todas estão funcionando
    print(f'{len(tickers)-1} Tickers elegíveis.')
    for ticker in tickers:
        try:
            html_dict[ticker] = general_functions.fetch_html(f"https://www.dadosdemercado.com.br/acoes/{ticker}")
            print(f'{len(html_dict.keys())} de {len(tickers)-1} HTMLs recumerados.')
        except:
            continue
    
    tickers = list(html_dict.keys())
    tickers_sa = general_functions.include_sa(tickers)
    infos = general_functions.get_info(tickers_sa)
    selic = general_functions.get_selic(selic_date)

    for ticker in tickers:
        # Construindo dataframes com as informações necessárias
        fin = finance_functions(ticker)
        df_balanco, df_dre, df_fluxo_caixa = fin.get_financial_dataframes(html_dict[ticker])

        # Formatando os dataframes
        df_balanco = fin.format_df_balanco(df_balanco)
        df_dre = fin.format_df_dre(df_dre)
        df_fluxo_caixa = fin.format_df_fluxo(df_fluxo_caixa)

        dfs_list = [df_balanco, df_dre, df_fluxo_caixa]
        try:
            dfs_list = [fin.format_dataframe(df) for df in dfs_list]
        except Exception as e:
            print(f'Valores inconsistenmtes para o cálculo de {ticker}: {str(e)}')
            continue
        df_balanco, df_dre, df_fluxo_caixa = dfs_list

        # Calculando o valor da ação e iniciando a contrução do output final
        val_acao = fin.Val_Acao(df_balanco, df_dre, df_fluxo_caixa, infos.tickers[f'{ticker}.SA'].info, selic)
        output[f'{ticker}.SA'] = [ticker]
        output[f'{ticker}.SA'].append(f'Valor calculado da ação: {val_acao}')

    # Descartando tickers não calculados
    tickers_sa = list(output.keys())
    print(f'{len(tickers_sa)-1} Tickers com valuation calculado.')

    # Baixar dados do Yahoo Finance
    try:
        data = general_functions.yf_download(tickers_sa, start_date, end_date)
    except Exception as e:
        print(f"Erro na API - {str(e)}")

    # Filtra os dados para manter apenas delta positivo
    weight_list = []
    for column in data.columns:
        try:
            calc = float(output[column][1].split(': ',2)[1])
            val = float(data[column].iloc[-1])
            if val < calc:
                weight_list.append(column)
        except:
            continue

    data = data[weight_list]

    # Verificar se há dados suficientes
    if len(data) < 2:
        print("Dados insuficientes para cálculo")
        return
    
    try:
        # Calcular pesos
        missing_tickers = [ticker for ticker in weight_list if ticker not in data.columns]
        if missing_tickers:
            print(f"Os seguintes tickers não possuem dados - {', '.join(missing_tickers)}")
        weights = econometria_functions.minvariance(data, 0.005)

        # Formatar o output final
        for coluna, weight in zip(data.columns, weights):
            output[coluna].append(f'Último valor de fechamento: {data[coluna].iloc[-1].round(4)}')
            output[coluna].append(f'Peso ótimo: {weight.round(4)}')
            print(f'{output[coluna]}')

    except Exception as e:
        print(f"Erro no cálculo - {str(e)}")
    


if __name__ == '__main__':
    main()