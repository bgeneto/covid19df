import contextlib
import datetime
import os
import pickle
import re
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable

import numpy as np
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
__version__ = "1.1.1"
__status__ = "Development"
__date__ = "20220309"


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
                response = requests.get(url, timeout=10)
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
        for t in range(1, 4):
            try:
                reqs = requests.get(url, timeout=10)
                soup = BeautifulSoup(reqs.text, 'html.parser')
                break
            except:
                continue
        else:
            display.fatal(
                "O site não está disponível no momento. Tente depois ")
            stop()

    # grab all pdf links
    hrefs = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and 'pdf' in href:
            hrefs.append(href)

    # filter pdf links by date
    urls = {}
    for href in hrefs:
        m = re.search(
            r"Obito.*?Covid.*?Notificado.*?(?P<dia>\d{2})[^0-9A-Za-z](?P<mes>\d{2})[^0-9A-Za-z](?P<ano>\d{2}).*\.pdf",
            href,
            flags=re.IGNORECASE
        )
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
    gen = None
    age = None
    vac = None
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as handle:
            gen = pickle.load(handle)
            age = pickle.load(handle)
            vac = pickle.load(handle)

    return gen, age, vac


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


def find_date(txt):
    '''Find notification date in txt file'''
    dtregex = r"^Notificado.*?em.*?(?P<dia>\d{2})[.-/](?P<mes>\d{2})[.-/](?P<ano>\d{4})"
    m = re.search(
        dtregex,
        txt,
        flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    # date found
    dia = m.group('dia')
    mes = m.group('mes')
    ano = m.group('ano')
    dt = f"{dia}-{mes}-{ano}"

    return dt


def scrap_age(all_txt_files):
    # find and load data to variable
    lst = []
    for f in all_txt_files:
        row = {}
        txt = Path(f.resolve()).read_text()

        # find notification date in file
        dt = find_date(txt)
        if not dt:
            display.error(f"Não foi possível ler a data no arquivo {f.stem}")
            continue
        row['data'] = dt

        # find única
        faixas = {'Menor de 2': '0 a 2',
                  '2 a 10': '2 a 10', '10 a 19': '10 a 19', '20 a 29': '20 a 29',
                  '30 a 39': '30 a 39', '40 a 49': '40 a 49', '50 a 59': '50 a 59',
                  '60 a 69': '60 a 69', '70 a 79': '70 a 79', '80 ou mais': '80+'}
        for f, nm in faixas.items():
            m = re.search(
                r"^%s.+?(?P<num>\d{1,2})" % f,
                txt,
                flags=re.IGNORECASE | re.MULTILINE)
            if not m:
                continue
            row[nm] = int(m.group('num'))
        # append inside first loop
        lst.append(row)

    df = pd.DataFrame(lst)
    if df.empty:
        display.fatal(
            "Não foi possível coletar os dados sobre as faixas etárias")
        stop()

    df.set_index('data', inplace=True)

    return df


def scrap_gen(all_txt_files):
    # find and load data to variable
    lst = []
    for f in all_txt_files:
        row = {}
        txt = Path(f.resolve()).read_text()

        # find notification date in file
        dt = find_date(txt)
        if not dt:
            display.error(f"Não foi possível ler a data no arquivo {f.stem}")
            continue
        row['data'] = dt

        # find deaths by gender
        m = re.search(
            r"^Feminino\s*(?P<feminino>\d+)\s*\nMasculino\s*(?P<masculino>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            display.error(
                f"Não foi possível contabilizar os óbitos por gênero na data {dt}")
            continue
        row['feminino'] = int(m.group('feminino'))
        row['masculino'] = int(m.group('masculino'))

        # finally we store all the data in a list of dictionaries
        lst.append(row)

    df = pd.DataFrame(lst)
    if df.empty:
        display.fatal("Não foi possível coletar os dados de óbitos por gênero")
        stop()

    df.set_index('data', inplace=True)

    return df


def scrap_vac(all_txt_files):
    # find and load data to variable
    lst = []
    for f in all_txt_files:
        errors = 0
        err_msg = ''
        row = {}
        txt = Path(f.resolve()).read_text()

        # find notification date in file
        dt = find_date(txt)
        if not dt:
            display.error(f"Não foi possível ler a data no arquivo {f.stem}")
            continue
        row['data'] = dt

        # find primeira
        m = re.search(
            r"^.+?primeira\s*dose\s*(?P<primeira>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg += f"Não foi possível contabilizar os óbitos de primeira dose na data {dt}"
            errors += 1
        else:
            row['primeira'] = int(m.group('primeira'))

        # find segunda
        m = re.search(
            r"^Segunda\s*Dose\s*(?P<segunda>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg += f"Não foi possível contabilizar os óbitos de segunda dose na data {dt}"
            errors += 1
        else:
            row['segunda'] = int(m.group('segunda'))

        # find única
        m = re.search(
            r"^Dose\s*[ÚúuU]nica\s*(?P<unica>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg += f"Não foi possível contabilizar os óbitos de dose única na data {dt}"
            errors += 1
        else:
            row['única'] = int(m.group('unica'))

        # find reforço
        m = re.search(
            r"^Dose.+?Refor[cçCÇ]o\s*(?P<reforco>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg += f"Não foi possível contabilizar os óbitos de dose de reforço na data {dt}"
            errors += 1
        else:
            row['reforço'] = int(m.group('reforco'))

        # find não vacinado
        m = re.search(
            r"^Não\s*vacinado\s*(?P<nenhuma>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg += f"Não foi possível contabilizar os óbitos de não vacinados na data {dt}"
            errors += 1
        else:
            row['nenhuma'] = int(m.group('nenhuma'))

        # find sem informação
        m = re.search(
            r"^Sem\s*informa.+?(?P<sem_info>\d+)",
            txt,
            flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            err_msg = f"Não foi possível contabilizar os óbitos sem informação de vacinação na data {dt}  \n"
            errors += 1
        else:
            row['sem info'] = int(m.group('sem_info'))

        # check errors
        if errors > 2:
            #    display.error(
            #        f"**ATENÇÃO: Não há dados sobre o esquema vacinal dos óbitos notificados no boletim do dia {dt}**")
            continue

        # finally we store all the data in a list of dictionaries
        lst.append(row)

    df = pd.DataFrame(lst)
    if df.empty:
        display.fatal("Não foi possível coletar os dados sobre a vacinação")
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
    #sidebar = initial_sidebar_config()

    url = 'https://www.saude.df.gov.br/boletinsinformativos-divep-cieves/'

    '''Desde o dia 1º de fevereiro de 2022, os boletins epidemiológicos
    da Secretaria de Saúde do Distrito Federal começaram a trazer
    a informação sobre o esquema vacinal dos óbitos notificados e registrados.'''

    '''Os dados apresentados abaixo foram extraídos a partir
       de 1º de fevereiro e podem ser conferidos na "Síntese diária de óbitos notificados" no endereço:'''

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
    gen, age, vac = load_cache(cache_file)

    # check last date in cache
    last_date_cached = None
    if gen is not None:
        last_date_cached = gen.index.max()
        try:
            last_date_cached = datetime.datetime.strptime(
                last_date_cached, "%d-%m-%Y").date()
        except:
            display.fatal("Formato de data inválido")
            stop()

    # check if cached needs refreshing
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    if not last_date_cached:
        cached = False
    else:
        cached = True if last_date_cached >= yesterday else False

    # download and convert only if not in cache
    if not cached:
        urls = get_links(url)
        with st.spinner('Baixando os relatórios...'):
            download_pdfs(urls)
            convert_to_txt()
        all_txt_files = [f for f in Path(output_dir).glob('*.txt')]
        gen = scrap_gen(all_txt_files)
        age = scrap_age(all_txt_files)
        vac = scrap_vac(all_txt_files)
        # cache new data
        with open(cache_file, 'wb') as handle:
            pickle.dump(gen, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(age, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(vac, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # sort by date (index is str not datetime)
    def sfunc(x): return x.str[6:10]+x.str[3:5]+x.str[0:2]
    gen.sort_index(
        inplace=True, key=sfunc)
    vac.sort_index(
        inplace=True, key=sfunc)
    age.sort_index(
        inplace=True, key=sfunc)
    age.sort_index(axis=1, inplace=True)

    # vac validation based on gender numbers
    for idx in gen.index:
        if idx in vac.index:
            diff = int(vac.loc[idx].sum()) - gen.loc[idx].sum()
            if diff < 0:
                vac.loc[idx] = np.nan
                display.warning(
                    f"Faltam dados sobre óbitos de vacinados no dia {idx}")

    # age validation based on gender numbers
    for idx in gen.index:
        if idx in age.index:
            diff = int(age.loc[idx].sum()) - gen.loc[idx].sum()
            if diff < 0:
                age.loc[idx, 'demais'] = abs(diff)
                display.warning(
                    f"Óbitos por faixa-etária corrigido para o dia {idx}")

    # soma
    sgen = gen.sum().rename('óbitos').to_frame()
    sage = age.sum().rename('óbitos').to_frame()
    svac = vac.sum().rename('óbitos').to_frame()

    # add percent column
    sgen['percentual'] = round(100.*sgen['óbitos']/sgen['óbitos'].sum(), 1)
    sage['percentual'] = round(100.*sage['óbitos']/sage['óbitos'].sum(), 1)
    svac['percentual'] = round(100.*svac['óbitos']/svac['óbitos'].sum(), 1)

    # special ordering
    sage.sort_index(
        inplace=True, key=lambda x: x.str[0:2])

    # rounded percents
    vacinados = round(svac.loc['única', 'percentual'] +
                      svac.loc['segunda', 'percentual'] +
                      svac.loc['reforço', 'percentual'], 1)
    nvacinados = round(svac.loc['nenhuma', 'percentual'], 1)
    incompleta = round(svac.loc['primeira', 'percentual'], 1)
    reforco = round(svac.loc['reforço', 'percentual'], 1)

    # display basic info
    display.info(f"O **percentual de óbitos** para os **vacinados** (dose única ou duas ou mais doses) \
        é de **{vacinados}%**")

    display.info(f"O **percentual de óbitos** para os **não-vacinados** (nenhuma dose) \
        é de **{nvacinados}%**")

    display.info(f"O **percentual de óbitos** para **vacinados com reforço** \
        é de **{reforco}%**")

    display.info(f"O **percentual de óbitos** para **vacinação incompleta** (somente 1ª dose) \
        é de **{incompleta}%**")

    display.info(f"O **total de óbitos** por covid-19 no DF entre 01-02-2022 e {gen.index.max()} \
        é de **{sgen['óbitos'].sum()}**")

    display.warning(
        f"ATENÇÃO: Os dados sobre o esquema vacinal dos óbitos foi removido dos boletins oficiais. \
          Tais dados estão disponíveis somente entre os dias 01-02-2022 e {vac.index.max()}")

    with st.expander("..:: DADOS COLETADOS ::.."):
        st.dataframe(
            pd.concat([df for df in [vac, gen, age]], axis=1, join='outer').reset_index())

    # ===========
    #    PLOTS
    # ===========

    # total de óbitos
    fig = px.bar(svac,
                 y=svac['óbitos'],
                 x=svac.index,
                 color=svac.index,
                 text=svac['percentual'].apply(lambda x: '{:.1f}%'.format(x)))
    fig.update_layout(
        title=dict(
            text="Total de Óbitos x Doses de Vacina",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="dose da vacina",
        yaxis_title="quantidade de óbitos",
        legend_title="dose vacina",
        hovermode="x",
        xaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # óbitos diários
    fig = px.bar(vac,
                 barmode="group",
                 text_auto=False)
    fig.update_layout(
        title=dict(
            text="Óbitos Diários x Doses de Vacina",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="data de notificação",
        yaxis_title="quantidade de óbitos",
        legend_title="dose vacina",
    )
    st.plotly_chart(fig, use_container_width=True)

    # total por idade
    fig = px.bar(sage,
                 color=sage.index,
                 y=sage['óbitos'],
                 x=sage.index,
                 text=sage['percentual'].apply(lambda x: '{:.1f}%'.format(x)))
    # fig.update_layout(bargap=0.2)
    fig.update_layout(
        title=dict(
            text="Total de Óbitos por Idade",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="faixa-etária",
        yaxis_title="quantidade de óbitos",
        legend_title="faixa-etária",
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)

    # diário por idade
    fig = px.bar(age,
                 barmode="group",
                 text_auto=False)
    fig.update_layout(
        title=dict(
            text="Óbitos Diários por Idade",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="data de notificação",
        yaxis_title="quantidade de óbitos",
        legend_title="faixa-etária",
    )
    st.plotly_chart(fig, use_container_width=True)

    # total por gênero
    fig = px.bar(sgen,
                 color=sgen.index,
                 y=sgen['óbitos'],
                 x=sgen.index,
                 text=sgen['percentual'].apply(lambda x: '{:.1f}%'.format(x)))
    # fig.update_layout(bargap=0.2)
    fig.update_layout(
        title=dict(
            text="Total de Óbitos por Gênero",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="gênero",
        yaxis_title="quantidade de óbitos",
        showlegend=False,
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)

    # diário por gênero
    fig = px.bar(gen,
                 barmode="group",
                 text_auto=False)
    # fig.update_layout(bargap=0.2)
    fig.update_layout(
        title=dict(
            text="Óbitos Diários por Gênero",
            x=0.0,
            y=0.925,
            font=dict(
                size=20,
            )
        ),
        xaxis_title="data de notificação",
        yaxis_title="quantidade de óbitos",
        legend_title="gênero",
    )
    st.plotly_chart(fig, use_container_width=True)

    # copyright, version and running time info
    st.caption(
        f":copyright: 2022 bgeneto | Versão: {__version__}")


if __name__ == '__main__':
    # always run as a streamlit app
    force_streamlit = True

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
