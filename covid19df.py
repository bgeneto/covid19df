import contextlib
import datetime
import os
import pickle
import re
import sys
from dis import dis
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable

import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from soupsieve import match

__author__ = "Bernhard Enders"
__maintainer__ = "Bernhard Enders"
__email__ = "b g e n e t o @ g m a i l d o t c o m"
__copyright__ = "Copyright 2022, Bernhard Enders"
__license__ = "GPL"
__version__ = "1.0.0"
__status__ = "Development"
__date__ = "20220208"


class Output:
    """Style our output"""

    def __init__(self) -> None:
        self.prefix = ""
        self.emoji = ""
        self.function = print
        if st._is_running_with_streamlit:
            self.function = st.write

    def msg(self, msg: str) -> None:
        if self.emoji:
            self.prefix = ""
        msg = str(self.emoji + self.prefix + str(msg)).strip()
        self.function(msg)

    def fatal(self, msg: str) -> None:
        self.prefix = " FATAL: "
        if st._is_running_with_streamlit:
            self.function = st.error
            self.emoji = " :x: "
        self.msg(msg)

    def debug(self, msg: str) -> None:
        self.prefix = " DEBUG: "
        if st._is_running_with_streamlit:
            self.function = st.write
            self.emoji = " :bug: "
        self.msg(msg)

    def error(self, msg: str) -> None:
        self.prefix = " ERRO: "
        if st._is_running_with_streamlit:
            self.function = st.error
        self.emoji = " :red_circle: "
        self.msg(msg)

    def warning(self, msg: str) -> None:
        self.prefix = " AVISO: "
        if st._is_running_with_streamlit:
            self.function = st.warning
            self.emoji = " :warning: "
        self.msg(msg)

    def info(self, msg: str) -> None:
        self.prefix = " INFO: "
        if st._is_running_with_streamlit:
            self.function = st.info
            self.emoji = " :bulb: "
        self.msg(msg)

    def success(self, msg: str) -> None:
        if st._is_running_with_streamlit:
            self.function = st.success
            self.emoji = " :white_check_mark: "
        self.msg(msg)

    def check(self, msg: str) -> None:
        if st._is_running_with_streamlit:
            self.function = st.write
            self.emoji = " :ballot_box_with_check: "
        self.msg(msg)

    def wait(self, msg: str) -> None:
        if st._is_running_with_streamlit:
            self.function = st.write
            self.emoji = " :stopwatch: "
        self.msg(msg)

    def write(self, msg) -> None:
        if st._is_running_with_streamlit:
            self.function = st.write
        self.function(msg)


def stop(code=0):
    if st._is_running_with_streamlit:
        st.stop()
    sys.exit(code)


def st_layout(title: str = "Streamlit App") -> None:
    """Configure Streamlit page layout"""

    st.set_page_config(
        page_title=title.split('-')[0],
        page_icon=":syringe:",
        layout="wide")
    st.title(title)
    # hide main (top right) menu
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def pdf_to_text(fn):
    '''pdf to text using pdfplumber'''
    text = ''
    with pdfplumber.open(fn) as pdf:
        for pdf_page in pdf.pages:
            text = text + pdf_page.extract_text(x_tolerance=1) + "\n"

    return text


def pdf_to_txt(fn):
    '''pdf to text using pdfminer'''
    from io import StringIO

    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    output_string = StringIO()
    with open(fn, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


@contextlib.contextmanager
def nostdout():
    from io import BytesIO
    save_stdout = sys.stdout
    sys.stdout = BytesIO()
    yield
    sys.stdout = save_stdout


def initial_sidebar_config():
    # sidebar contents
    sidebar = st.sidebar
    sidebar.subheader("..:: MENU ::..")

    return sidebar


def download_pdfs(urls: dict) -> int:
    # download pdfs files
    num_pdfs = 0
    for dt, url in urls.items():
        fn = os.path.join(
            output_dir, f"Resumo_Obito_Covid_Notificados_{dt}.pdf")
        # download only if file not found
        if os.path.isfile(fn):
            continue
        for t in range(1, 4):
            try:
                response = requests.get(url)
                break
            except:
                continue
        else:
            display.error(
                f"Não foi possivel baixar o arquivo pdf da seguinte data: {dt}")
        try:
            with open(fn, 'wb') as f:
                f.write(response.content)
            num_pdfs += 1
        except:
            display.error(f"Não foi possível gravar o arquivo pdf {fn}")

    if num_pdfs > 0:
        display.info(f"{num_pdfs} arquivo(s) baixado(s) com sucesso")
    return num_pdfs


def get_links(url) -> dict:
    with st.spinner('Aguarde, conectando ao site da Secretaria de Saúde...'):
        try:
            reqs = requests.get(url)
            soup = BeautifulSoup(reqs.text, 'html.parser')
        except:
            display.fatal("A URL informada não está disponível no momento")
            stop()

    # grab all pdf links
    hrefs = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and 'pdf' in href:
            hrefs.append(href)

    # initial date that has vaccination info
    iday = '01'
    imonth = '02'
    iyear = '22'

    # filter pdf links by date
    urls = {}
    for href in hrefs:
        m = re.search(
            r"Obito.*?Covid.*?Notificado.*?(?P<dia>\d{2})[^0-9A-Za-z](?P<mes>\d{2})[^0-9A-Za-z](?P<ano>\d{2})\.pdf", href, flags=re.IGNORECASE)
        if m is not None:
            try:
                dia = m.group('dia')
                mes = m.group('mes')
                ano = m.group('ano')
                dt = f"{dia}-{mes}-{ano}"
                if (int(mes) >= 2) and (int(ano) >= 22):
                    urls[dt] = href
            except:
                display.warning(
                    f"AVISO: Não foi possível identificar a data na URL {href}")

    # check number of pdf files found
    if len(urls) < 1:
        display.fatal("Nenhum link contendo as palavras-chave foi localizado")
        stop()

    return urls


def load_cache(cache_file: str) -> pd.DataFrame:
    df = None
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as handle:
            df = pickle.load(handle)

    return df


def convert_to_txt() -> int:
    # convert pdf to txt format
    all_pdf_files = [f for f in Path(output_dir).glob('*.pdf')]
    num_txt_files = 0
    for f in all_pdf_files:
        fn = os.path.join(output_dir, f.stem+'.txt')
        # convert only if file not found
        if os.path.isfile(fn):
            continue
        text = pdf_to_text(f.resolve())
        try:
            with open(fn, 'w') as txt:
                txt.write(text)
            num_txt_files += 1
        except:
            display.error(f"Não foi possível gravar o arquivo txt {fn}")

    if num_txt_files > 0:
        display.info(f"{num_txt_files} arquivo(s) convertido(s) com sucesso")
    return num_txt_files


def scrap_data():
    # find and load data to variable
    all_txt_files = [f for f in Path(output_dir).glob('*.txt')]
    lst = []
    for f in all_txt_files:
        row = {}
        txt = Path(f.resolve()).read_text()
        # find date
        m = re.search(
            r"^Notificado.*?em.*?(?P<dia>\d{2})[^0-9A-Za-z](?P<mes>\d{2})[^0-9A-Za-z](?P<ano>\d{4})$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(f"Não foi possível ler a data no arquivo {f.stem}")
            continue
        dia = m.group('dia')
        mes = m.group('mes')
        ano = m.group('ano')
        dt = f"{dia}-{mes}-{ano}"
        row['data'] = f"{dia}/{mes}/{ano}"

        # find deaths by gender
        m = re.search(
            r"^Feminino\s*(?P<feminino>\d+)\nMasculino\s*(?P<masculino>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos por gênero na data {dt}")
            continue
        row['feminino'] = int(m.group('feminino'))
        row['masculino'] = int(m.group('masculino'))

        # find primeira
        m = re.search(
            r"^.+?primeira\s*dose\s*(?P<primeira>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos de primeira dose na data {dt}")
            continue
        row['primeira'] = int(m.group('primeira'))

        # find segunda
        m = re.search(
            r"^Segunda\s*Dose\s*(?P<segunda>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos de segunda dose na data {dt}")
            continue
        row['segunda'] = int(m.group('segunda'))

        # find única
        m = re.search(
            r"^Dose\s*[ÚúuU]nica\s*(?P<unica>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos de dose única na data {dt}")
            continue
        row['única'] = int(m.group('unica'))

        # find reforço
        m = re.search(
            r"^Dose.+?Refor[cçCÇ]o\s*(?P<reforco>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos de dose de reforço na data {dt}")
            continue
        row['reforço'] = int(m.group('reforco'))

        # find não vacinado
        m = re.search(
            r"^Não\s*vacinado\s*(?P<nenhuma>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos de não vacinados na data {dt}")
            continue
        row['nenhuma'] = int(m.group('nenhuma'))

        # find sem informação
        m = re.search(
            r"^Sem\s*informa.+?(?P<sem_info>\d+)$",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos sem informação de vacinação na data {dt}")
            continue
        row['sem info'] = int(m.group('sem_info'))

        # finally we store all the data in a list of dictionaries
        lst.append(row)

    df = pd.DataFrame(lst)
    if df.empty:
        display.fatal("Não foi possível coletar os dados")
        stop()

    df.set_index('data', inplace=True)
    #df.index = pd.DatetimeIndex(df.index)

    return df


def check_num_files(cache_file):
    num_txt_files = [f for f in Path(output_dir).glob('*.txt')]
    num_pdf_files = [f for f in Path(output_dir).glob('*.pdf')]
    if (len(num_txt_files) != len(num_pdf_files)):
        display.warning(
            "Número inconsistente de arquivos")
        try:
            os.remove(cache_file)
        except:
            pass


def main():
    start = timer()

    #sidebar = initial_sidebar_config()

    '''Desde o dia 1º de fevereiro de 2022, os boletins epidemiológicos
    da Secretaria de Saúde do Distrito Federal começaram a trazer
    a informação sobre o esquema vacinal dos óbitos notificados e registrados.'''

    '''Os dados apresentados abaixo foram extraídos a partir
       de 1º de fevereiro e podem ser conferidos em:'''

    url = 'https://www.saude.df.gov.br/boletinsinformativos-divep-cieves/'

    display.write(url)

    # create output dir
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            display.fatal("Não foi possível criar o diretório de saída!")
            stop()

    # save data in cache file
    cache_file = os.path.join(output_dir, 'dados.pickle')

    # check inconsistency in converted files
    check_num_files(cache_file)

    # load cached raw_data (deserialize)
    df = load_cache(cache_file)

    # check last date in cache
    last_date_cached = None
    if df is not None:
        last_date_cached = df.index.max()
        try:
            last_date_cached = datetime.datetime.strptime(
                last_date_cached, "%d/%m/%Y").date()
        except:
            display.fatal("Formato de data inválido")
            stop()

    # check if cached needs refreshing
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    cached = True if last_date_cached == yesterday else False

    # download and convert only if not in cache
    if not cached:
        urls = get_links(url)
        with st.spinner('Baixando os relatórios...'):
            download_pdfs(urls)
            convert_to_txt()
        df = scrap_data()
        # cache new data
        with open(cache_file, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # quick data validation
    for dt, sr in df.iterrows():
        soma_gen = sr[0:2].sum()
        soma_vac = sr[2:].sum()
        if soma_gen != soma_vac:
            display.warning(f"Inconsistência detectada nos dados de {dt}")

    sdf = df.sum().rename('valor').to_frame()
    num_total_obitos = sdf.loc['feminino',
                               'valor'] + sdf.loc['masculino', 'valor']
    sdf['percentual'] = round(100.*sdf['valor']/num_total_obitos, 1)

    vacinados = round(sdf.loc['segunda', 'percentual'] +
                      sdf.loc['reforço', 'percentual'], 1)
    nvacinados = sdf.loc['nenhuma', 'percentual']
    nvacinados_max = round(sdf.loc['nenhuma', 'percentual'] +
                           sdf.loc['sem info', 'percentual'], 1)

    display.info(f"O **percentual de óbitos** entre os **vacinados** (com duas ou mais doses) \
        é de **{vacinados}%**")

    display.info(f"O **percentual de óbitos** entre os **não-vacinados** (nenhuma dose) \
        é de **{nvacinados}%**, podendo chegar a {nvacinados_max}%")

    last_date = yesterday if not last_date_cached else last_date_cached
    display.warning(
        f"OBS.: considerando os dados disponíveis até o dia {last_date.strftime('%m/%d/%Y')}")

    with st.expander("..:: DADOS COLETADOS ::.."):
        display.write(df)

    # ===========
    #    PLOTS
    # ===========

    # total de óbitos
    ndf = sdf.iloc[2:]
    fig = px.bar(ndf,
                 y=ndf['valor'],
                 x=ndf.index,
                 color=ndf.index,
                 text=ndf['percentual'].apply(lambda x: '{:.1f}%'.format(x)))
    fig.update_layout(
        title_text="Histórico de Óbitos x Doses de Vacina",
        title_x=0.5,
        xaxis_title="dose da vacina",
        yaxis_title="quantidade",
    )
    st.plotly_chart(fig, use_container_width=True)

    # óbitos diários
    ndf = df.iloc[:, 2:]
    fig = px.bar(ndf,
                 barmode="group",
                 text_auto=True)
    fig.update_layout(
        title_text="Óbitos Diários x Doses de Vacina",
        title_x=0.5,
        xaxis_title="dia",
        yaxis_title="quantidade",
        legend_title="dose vacina",
    )
    st.plotly_chart(fig, use_container_width=True)

    # total por gênero
    ndf = sdf.iloc[:2]
    fig = px.bar(ndf,
                 color=ndf.index,
                 y=ndf['valor'],
                 x=ndf.index,
                 text=ndf['percentual'].apply(lambda x: '{:.1f}%'.format(x)))
    # fig.update_layout(bargap=0.2)
    fig.update_layout(
        title_text="Histórico de Óbitos por Gênero",
        title_x=0.5,
        xaxis_title="gênero",
        yaxis_title="quantidade",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # diário por gênero
    ndf = df.iloc[:, :2]
    fig = px.bar(ndf,
                 #y=['masculino', 'feminino'],
                 barmode="group",
                 text_auto=True)
    # fig.update_layout(bargap=0.2)
    fig.update_layout(
        title_text="Óbitos Diários por Gênero",
        title_x=0.5,
        xaxis_title="dia",
        yaxis_title="quantidade",
        legend_title="gênero",
    )
    st.plotly_chart(fig, use_container_width=True)

    # copyright, version and running time info
    end = timer()
    st.caption(
        f":copyright: 2022 bgeneto | Version: {__version__} | Execution time: {(end-start):.2f}")


if __name__ == '__main__':
    # always run as a streamlit app
    force_streamlit = False

    output_dir = 'data'

    # page title/header
    title = "Óbitos de Vacinados no DF"

    # configure print output (streamlit, python, ipython etc...)
    display = Output()

    # check if running as standalone python script or via streamlit
    if st._is_running_with_streamlit:
        st_layout(title)
        main()
    else:
        if force_streamlit:
            st_layout(title)
            from streamlit import cli as stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
        else:
            main()