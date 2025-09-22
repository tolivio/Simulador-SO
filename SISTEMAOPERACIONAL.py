#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from enum import Enum, auto
from collections import deque
from typing import Optional, List, Dict
import argparse, json

# ======== PARÂMETROS PADRÃO (editar se necessário) ========
ESCALONADOR_PADRAO = "fcfs"   # "fcfs", "rr", "prio"
QUANTUM_PADRAO = 4
PAGINA_PADRAO = 256
MOLDURAS_PADRAO = 64
DISPOSITIVOS_PADRAO = {"disk": 10, "tty": 2}

# -------------------------- Estados e Estruturas --------------------------

class EstadoThread(Enum):
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    BLOCKED = auto()
    TERMINATED = auto()

class PCB:  # bloco de controle de processo (informações do processo)
    def __init__(self, pid:int, prioridade:int=0):
        self.pid = pid
        self.prioridade = prioridade
        self.arquivos: Dict[str, object] = {}
        self.tabela_paginas: Dict[int, int] = {}
        self.tempo_criacao = 0
        self.tempo_termino: Optional[int] = None

class TCB:  # bloco de controle de thread (cada thread do processo)
    _contador_tid = 1
    def __init__(self, processo:'Processo', operacoes:List[str]):
        self.tid = TCB._contador_tid; TCB._contador_tid += 1
        self.processo = processo
        self.estado = EstadoThread.NEW
        self.pc = 0
        self.operacoes = list(operacoes)
        self.primeira_resposta: Optional[int] = None

    def operacao_atual(self) -> Optional[str]:
        return self.operacoes[self.pc] if self.pc < len(self.operacoes) else None

# -------------------------- Métricas e Registro --------------------------

class Metricas:
    def __init__(self):
        self.trocas_contexto = 0
        self.sobre_carga_escalonador = 0
        self.cpu_ocupada = 0
        self.dispositivos_ocupacao: Dict[str,int] = {}
        self.falhas_pagina = 0
        self.estatisticas_threads: Dict[int, Dict[str, Optional[int]]] = {}

    def garantir(self, tid:int):
        self.estatisticas_threads.setdefault(tid, dict(cpu=0, ready=0, resposta=None))

    def tick_cpu(self, tid:int):
        self.cpu_ocupada += 1
        self.garantir(tid); self.estatisticas_threads[tid]['cpu'] += 1

    def tick_pronto(self, tid:int):
        self.garantir(tid); self.estatisticas_threads[tid]['ready'] += 1

    def registrar_resposta(self, tid:int, clock:int):
        self.garantir(tid)
        if self.estatisticas_threads[tid]['resposta'] is None:
            self.estatisticas_threads[tid]['resposta'] = clock

    def relatorio(self, clock:int, processos:List['Processo']):
        tempos_turnaround = []
        espera_pronto = []
        for p in processos:
            if p.pcb.tempo_termino is not None:
                tempos_turnaround.append(p.pcb.tempo_termino - p.pcb.tempo_criacao)
        for s in self.estatisticas_threads.values():
            espera_pronto.append(s['ready'])
        util_cpu = (self.cpu_ocupada / clock) if clock > 0 else 0.0
        return {
            'tempo_simulacao': clock,
            'utilizacao_cpu': round(util_cpu,3),
            'trocas_contexto': self.trocas_contexto,
            'sobre_carga_escalonador': self.sobre_carga_escalonador,
            'falhas_pagina': self.falhas_pagina,
            'turnaround_medio': round(sum(tempos_turnaround)/len(tempos_turnaround),2) if tempos_turnaround else 0,
            'espera_pronto_media_ticks': round(sum(espera_pronto)/len(espera_pronto),2) if espera_pronto else 0
        }

class Registro:
    def __init__(self):
        self.linhas: List[str] = []
    def gravar(self, clock:int, msg:str):
        self.linhas.append(f"[{clock:06d}] {msg}")
    def salvar(self, caminho:str):
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.linhas))

# -------------------------- Escalonadores --------------------------

class EscalonadorBase:
    def __init__(self):
        self.prontos = deque()
    def enfileirar(self, t:TCB):
        t.estado = EstadoThread.READY
        self.prontos.append(t)
    def selecionar(self) -> Optional[TCB]:
        if not self.prontos: return None
        t = self.prontos.popleft(); t.estado = EstadoThread.RUNNING; return t
    def on_tick(self, so:'SO'): pass

class EscalonadorFCFS(EscalonadorBase):
    pass

class EscalonadorRR(EscalonadorBase):
    def __init__(self, quantum:int):
        super().__init__(); self.quantum = quantum; self.fatia = 0; self.atual = None
    def selecionar(self):
        self.fatia = 0
        self.atual = super().selecionar()
        return self.atual
    def on_tick(self, so:'SO'):
        if self.atual and self.atual.estado == EstadoThread.RUNNING:
            self.fatia += 1
            if self.fatia >= self.quantum:
                so.preemptar_running("quantum RR")

class EscalonadorPrioridade(EscalonadorBase):
    def __init__(self, preemptivo=True):
        super().__init__(); self.preemptivo = preemptivo
    def enfileirar(self, t:TCB):
        t.estado = EstadoThread.READY; self.prontos.append(t)
        self.prontos = deque(sorted(self.prontos, key=lambda x: x.processo.pcb.prioridade))
    def on_tick(self, so:'SO'):
        if not self.preemptivo or not so.executando or not self.prontos: return
        if self.prontos[0].processo.pcb.prioridade < so.executando.processo.pcb.prioridade:
            so.preemptar_running("preempção por prioridade")

# -------------------------- Gerenciador de Memória (paginação) --------------------------

class GerenciadorMemoria:
    def __init__(self, tamanho_pagina:int=256, molduras:int=64, metricas:Optional[Metricas]=None):
        self.tamanho_pagina = tamanho_pagina
        self.molduras = molduras
        self.livres = list(range(molduras))
        self.metricas = metricas

    def acessar(self, pcb:PCB, endereco_logico:int):
        pagina = endereco_logico // self.tamanho_pagina
        if pagina not in pcb.tabela_paginas:
            self.falta_pagina(pcb, pagina)
        quadro = pcb.tabela_paginas[pagina]
        return quadro * self.tamanho_pagina + (endereco_logico % self.tamanho_pagina)

    def falta_pagina(self, pcb:PCB, pagina:int):
        if self.metricas: self.metricas.falhas_pagina += 1
        if self.livres:
            quadro = self.livres.pop(0)
        else:
            # substituição simples: remove o primeiro mapeamento do processo
            vitima = next(iter(pcb.tabela_paginas.keys()), 0)
            quadro = pcb.tabela_paginas.get(vitima, 0)
            pcb.tabela_paginas.pop(vitima, None)
        pcb.tabela_paginas[pagina] = quadro

# -------------------------- Dispositivos e Sistema de Arquivos --------------------------

class Dispositivo:
    def __init__(self, nome:str, tempo_servico:int, metricas:Metricas, registro:Registro):
        self.nome = nome
        self.tempo_servico = tempo_servico
        self.metricas = metricas
        self.fila = deque()
        self.atual = None
        self.restante = 0
        self.registro = registro

    def requisitar(self, thread:TCB, duracao:int):
        self.fila.append((thread, max(duracao,1)))

    def tick(self, clock:int, so:'SO'):
        if self.atual:
            self.metricas.dispositivos_ocupacao.setdefault(self.nome, 0)
            self.metricas.dispositivos_ocupacao[self.nome] += 1
            self.restante -= 1
            if self.restante <= 0:
                t,_ = self.atual
                self.registro.gravar(clock, f"IO completo {self.nome} T{t.tid}")
                so.desbloquear_thread(t)
                self.atual = None
        if not self.atual and self.fila:
            t,d = self.fila.popleft()
            self.atual = (t,d); self.restante = d
            self.registro.gravar(clock, f"IO início {self.nome} T{t.tid} ({d})")

class Arquivo:
    def __init__(self, nome:str):
        self.nome = nome; self.tamanho = 0; self.conteudo = b''

class SistemaArquivos:
    def __init__(self):
        self.estrutura = {'/': {}}
    def _split(self, caminho:str):
        return [p for p in caminho.split('/') if p]
    def criar(self, caminho:str):
        partes = self._split(caminho); d = self.estrutura['/']
        for p in partes[:-1]: d = d.setdefault(p, {})
        nome = partes[-1]; d[nome] = Arquivo(nome)
    def obter(self, caminho:str):
        partes = self._split(caminho); d = self.estrutura['/']
        for p in partes[:-1]: d = d.get(p, {})
        return d.get(partes[-1])
    def escrever(self, caminho:str, n:int):
        partes = self._split(caminho); d = self.estrutura['/']
        for p in partes[:-1]: d = d.setdefault(p, {})
        f = d.get(partes[-1])
        if not f:
            self.criar(caminho); f = d.get(partes[-1])
        f.tamanho += n; f.conteudo = (f.conteudo + b'X'*n)[-4096:]
    def ler(self, caminho:str, n:int):
        f = self.obter(caminho); return min(n, f.tamanho) if f else 0
    def listar(self, caminho:str='/'):
        d = self.estrutura['/']
        for p in self._split(caminho): d = d.get(p, {})
        return list(d.keys())

# -------------------------- SO (orquestrador) --------------------------

class Processo:
    _contador_pid = 1
    def __init__(self, prioridade:int=0):
        self.pcb = PCB(Processo._contador_pid, prioridade); Processo._contador_pid += 1
        self.threads: List[TCB] = []

class SO:
    def __init__(self, configuracao:dict, workload:dict):
        self.clock = 0
        self.metricas = Metricas()
        self.registro = Registro()
        self.processos: List[Processo] = []
        self.dispositivos: Dict[str, Dispositivo] = {}
        self.sistema_arquivos = SistemaArquivos()

        # escolher escalonador
        tipo = configuracao.get('sched', ESCALONADOR_PADRAO)
        if tipo == 'fcfs':
            self.escalonador = EscalonadorFCFS()
        elif tipo == 'rr':
            self.escalonador = EscalonadorRR(configuracao.get('quantum', QUANTUM_PADRAO))
        else:
            self.escalonador = EscalonadorPrioridade(True)

        # gerenciador de memória
        self.mem = GerenciadorMemoria(configuracao.get('pagesize', PAGINA_PADRAO),
                                      configuracao.get('frames', MOLDURAS_PADRAO),
                                      self.metricas)

        # dispositivos
        devs = configuracao.get('devices') or DISPOSITIVOS_PADRAO
        for nome, svc in devs.items():
            self.dispositivos[nome] = Dispositivo(nome, svc, self.metricas, self.registro)

        self.executando: Optional[TCB] = None
        self.carregar_workload(workload)

    def carregar_workload(self, wl:dict):
        for pdef in wl['processes']:
            p = Processo(prioridade=pdef.get('priority', 0))
            p.pcb.tempo_criacao = self.clock
            for tdef in pdef['threads']:
                t = TCB(p, tdef['ops'])
                p.threads.append(t)
                self.escalonador.enfileirar(t)
                self.registro.gravar(self.clock, f"NEW P{p.pcb.pid} T{t.tid}")
            self.processos.append(p)

    def despachar(self):
        prox = self.escalonador.selecionar()
        if prox:
            if prox.primeira_resposta is None:
                prox.primeira_resposta = self.clock
                self.metricas.registrar_resposta(prox.tid, self.clock)
            self.executando = prox
            self.metricas.trocas_contexto += 1
            self.registro.gravar(self.clock, f"DISPATCH T{prox.tid} (P{prox.processo.pcb.pid})")

    def preemptar_running(self, motivo:str):
        if self.executando:
            self.registro.gravar(self.clock, f"PREEMPT T{self.executando.tid}: {motivo}")
            self.metricas.sobre_carga_escalonador += 1
            self.escalonador.enfileirar(self.executando)
            self.executando = None

    def bloquear_running(self):
        if self.executando:
            self.registro.gravar(self.clock, f"BLOCK T{self.executando.tid}")
            self.executando.estado = EstadoThread.BLOCKED
            self.executando = None

    def desbloquear_thread(self, t:TCB):
        t.estado = EstadoThread.READY
        self.escalonador.enfileirar(t)
        self.registro.gravar(self.clock, f"UNBLOCK T{t.tid}")

    def tick(self):
        if not self.executando:
            self.despachar()

        # tick dos dispositivos
        for dev in self.dispositivos.values():
            dev.tick(self.clock, self)

        # contabiliza tempo em pronto (métrica)
        for t in list(self.escalonador.prontos):
            self.metricas.tick_pronto(t.tid)

        # executa 1 tick de CPU, se houver thread
        if self.executando:
            self.step_thread(self.executando)
            # checar se ainda está executando após step (pode ter terminado ou sido bloqueada)
            if self.executando:
                self.metricas.tick_cpu(self.executando.tid)
                self.escalonador.on_tick(self)

        self.clock += 1

    def step_thread(self, t:TCB):
        op = t.operacao_atual()
        if op is None:
            t.estado = EstadoThread.TERMINATED
            self.registro.gravar(self.clock, f"EXIT T{t.tid}")
            t.processo.pcb.tempo_termino = self.clock
            self.executando = None
            return

        tokens = op.split()
        cabecalho = tokens[0]

        if cabecalho.startswith('cpu:'):
            n = int(cabecalho.split(':')[1]); n -= 1
            if n <= 0:
                t.pc += 1
            else:
                t.operacoes[t.pc] = f"cpu:{n}"
            return

        if cabecalho.startswith('io_nb:'):
            _, resto = cabecalho.split(':',1); nome_dev, dur = resto.split(':'); dur = int(dur)
            dev = self.dispositivos.get(nome_dev)
            if dev: dev.requisitar(t, dur)
            t.pc += 1; return

        if cabecalho.startswith('io:'):
            _, resto = cabecalho.split(':',1); nome_dev, dur = resto.split(':'); dur = int(dur)
            dev = self.dispositivos.get(nome_dev)
            if dev:
                dev.requisitar(t, dur)
                self.bloquear_running()
            t.pc += 1; return

        if cabecalho.startswith('mem:'):
            endereco = int(cabecalho.split(':')[1])
            _ = self.mem.acessar(t.processo.pcb, endereco)
            t.pc += 1; return

        if cabecalho.startswith('fs:'):
            partes = cabecalho.split(':')
            sub = partes[1] if len(partes) > 1 else (tokens[1] if len(tokens) > 1 else '')
            resto = tokens[1:] if len(partes) == 1 else partes[2:] + tokens[1:]
            if sub == 'create':
                caminho = resto[0]; tamanho = int(resto[1]); self.sistema_arquivos.criar(caminho); self.sistema_arquivos.escrever(caminho, tamanho)
            elif sub == 'write':
                caminho = resto[0]; n = int(resto[1]); self.sistema_arquivos.escrever(caminho, n)
            elif sub == 'read':
                caminho = resto[0]; n = int(resto[1]); _ = self.sistema_arquivos.ler(caminho, n)
            t.pc += 1; return

        # operação desconhecida: avança
        t.pc += 1

    def finalizado(self):
        return all(t.estado == EstadoThread.TERMINATED for p in self.processos for t in p.threads)

    def executar(self, max_ticks:int=100000):
        while not self.finalizado() and self.clock < max_ticks:
            self.tick()
        self.registro.gravar(self.clock, "SIM END")
        return self.metricas.relatorio(self.clock, self.processos)

# -------------------------- Workload e CLI --------------------------

WORKLOAD_PADRAO = {
    "processes": [
        {
            "priority": 1,
            "threads": [
                {"ops": ["cpu:5", "io_nb:tty:3", "mem:1024", "io:disk:8", "fs:create /tmp/a.txt 128", "cpu:3"]},
                {"ops": ["cpu:4", "io:tty:4", "cpu:2"]}
            ]
        },
        {
            "priority": 2,
            "threads": [
                {"ops": ["cpu:2", "mem:512", "cpu:6", "fs:write /tmp/a.txt 64", "cpu:1"]}
            ]
        }
    ]
}

def parse_devices(args_devices:Optional[List[str]]):
    d = {}
    for arg in (args_devices or []):
        nome, svc = arg.split(':'); d[nome] = int(svc)
    return d or dict(DISPOSITIVOS_PADRAO)

def main():
    ap = argparse.ArgumentParser(description="Simulador de SO (arquivo único, nomes em PT-BR)")
    ap.add_argument('--sched', choices=['fcfs','rr','prio'], default=ESCALONADOR_PADRAO)
    ap.add_argument('--quantum', type=int, default=QUANTUM_PADRAO)
    ap.add_argument('--pagesize', type=int, default=PAGINA_PADRAO)
    ap.add_argument('--frames', type=int, default=MOLDURAS_PADRAO)
    ap.add_argument('--device', action='append', default=[])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workload-inline', type=str, default='')
    ap.add_argument('--logfile', type=str, default='')
    args = ap.parse_args()

    if args.seed:
        import random; random.seed(args.seed)

    cfg = {
        'sched': args.sched,
        'quantum': args.quantum,
        'pagesize': args.pagesize,
        'frames': args.frames,
        'devices': parse_devices(args.device)
    }

    wl = json.loads(args.workload_inline) if args.workload_inline else WORKLOAD_PADRAO

    so = SO(cfg, wl)
    rel = so.executar()

    if args.logfile:
        so.registro.salvar(args.logfile)
        print(f"[i] Registro salvo em: {args.logfile}")

    print("MÉTRICAS:", json.dumps(rel, indent=2))

if __name__ == '__main__':
    main()
    input("\nPressione Enter para sair...")
