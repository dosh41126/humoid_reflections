import tkinter as tk
import customtkinter
import threading
import os
import sqlite3 
import logging
import numpy as np
import base64
import queue
import uuid
import requests
import io
import sys
import random
import re
import json
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from os import path
from collections import Counter
from summa import summarizer
import nltk
from textblob import TextBlob
from weaviate.util import generate_uuid5
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from datetime import datetime
from weaviate.embedded import EmbeddedOptions
import weaviate
import pennylane as qml
import psutil
import webcolors
import colorsys
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type
import bleach
import httpx
import math
from typing import List, Tuple
from math import log2

ARGON2_TIME_COST_DEFAULT = 3          
ARGON2_MEMORY_COST_KIB    = 262144   
ARGON2_PARALLELISM        = max(1, min(4, os.cpu_count() or 1))
ARGON2_HASH_LEN           = 32
CRYSTALLIZE_THRESHOLD = 5    
DECAY_FACTOR = 0.95          
VAULT_PASSPHRASE_ENV      = "VAULT_PASSPHRASE"
VAULT_VERSION             = 1        
DATA_KEY_VERSION          = 1         
VAULT_NONCE_SIZE          = 12    
DATA_NONCE_SIZE           = 12
AGING_T0_DAYS = 7.0          
AGING_GAMMA_DAYS = 5.0       
AGING_PURGE_THRESHOLD = 0.5  
AGING_INTERVAL_SECONDS = 3600  
LAPLACIAN_ALPHA = 0.18 
JS_LAMBDA       = 0.10 

def _aad_str(*parts: str) -> bytes:
    return ("|".join(parts)).encode("utf-8")

customtkinter.set_appearance_mode("Dark")
nltk.data.path.append("/root/nltk_data")

def download_nltk_data():
    try:
        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
            'corpora/brown': 'brown',
            'corpora/wordnet': 'wordnet',
            'corpora/stopwords': 'stopwords',
            'corpora/conll2000': 'conll2000'
        }

        for path_, package in resources.items():
            try:
                nltk.data.find(path_)
                print(f"'{package}' already downloaded.")
            except LookupError:
                nltk.download(package)
                print(f"'{package}' downloaded successfully.")

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


download_nltk_data()
        
client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
executor = ThreadPoolExecutor(max_workers=5)
bundle_dir = path.abspath(path.dirname(__file__))
path_to_config = path.join(bundle_dir, 'config.json')
model_path = "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
mmproj_path = "/data/llama-3-vision-alpha-mmproj-f16.gguf"
logo_path = path.join(bundle_dir, 'logo.png')

def load_config(file_path=path_to_config):
    with open(file_path, 'r') as file:
        return json.load(file)

q = queue.Queue()
logger = logging.getLogger(__name__)
config = load_config()
SAFE_ALLOWED_TAGS: list[str] = []
SAFE_ALLOWED_ATTRS: dict[str, list[str]] = {}
SAFE_ALLOWED_PROTOCOLS: list[str] = []
_CONTROL_WHITELIST = {'\n', '\r', '\t'}

def _strip_control_chars(s: str) -> str:

    return ''.join(ch for ch in s if ch.isprintable() or ch in _CONTROL_WHITELIST)

def sanitize_text(
    text: str,
    *,
    max_len: int = 4000,
    strip: bool = True,
) -> str:

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text[:max_len]
    text = _strip_control_chars(text)
    cleaned = bleach.clean(
        text,
        tags=SAFE_ALLOWED_TAGS,
        attributes=SAFE_ALLOWED_ATTRS,
        protocols=SAFE_ALLOWED_PROTOCOLS,
        strip=strip,
        strip_comments=True,
    )
    return cleaned

_PROMPT_INJECTION_PAT = re.compile(
    r'(?is)(?:^|\n)\s*(system:|assistant:|ignore\s+previous|do\s+anything|jailbreak\b).*'
)

def sanitize_for_prompt(text: str, *, max_len: int = 2000) -> str:

    cleaned = sanitize_text(text, max_len=max_len)
    cleaned = _PROMPT_INJECTION_PAT.sub('', cleaned)
    return cleaned.strip()

def sanitize_for_graphql_string(s: str, *, max_len: int = 512) -> str:

    s = sanitize_text(s, max_len=max_len)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return s

DB_NAME = config['DB_NAME']
API_KEY = config['API_KEY']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_QUERY_PATH = config['WEAVIATE_QUERY_PATH']

class SecureEnclave:

    def __enter__(self):
        self._buffers = []
        return self

    def track(self, buf):
        self._buffers.append(buf)
        return buf

    def __exit__(self, exc_type, exc, tb):
        for b in self._buffers:
            try:
                if isinstance(b, np.ndarray):
                    b.fill(0.0)
            except Exception:
                pass
        self._buffers.clear()

class AdvancedHomomorphicVectorMemory:

    AAD_CONTEXT = _aad_str("fhe", "embeddingv2")
    DIM = 64
    QUANT_SCALE = 127.0  

    def __init__(self):

        master_key = crypto._derived_keys[crypto.active_version]
        seed = int.from_bytes(hashlib.sha256(master_key).digest()[:8], "big")
        rng = np.random.default_rng(seed)

        A = rng.normal(size=(self.DIM, self.DIM))
        Q, _ = np.linalg.qr(A)
        self.rotation = Q  

        self.lsh_planes = rng.normal(size=(16, self.DIM)) 

    def _rotate(self, vec: np.ndarray) -> np.ndarray:
        return self.rotation @ vec

    def _quantize(self, vec: np.ndarray) -> list[int]:
        clipped = np.clip(vec, -1.0, 1.0)
        return (clipped * self.QUANT_SCALE).astype(np.int8).tolist()

    def _dequantize(self, q: list[int]) -> np.ndarray:
        arr = np.array(q, dtype=np.float32) / self.QUANT_SCALE
        return arr

    def _simhash_bucket(self, rotated_vec: np.ndarray) -> str:
        dots = self.lsh_planes @ rotated_vec
        bits = ["1" if d >= 0 else "0" for d in dots]
        return "".join(bits) 

    def encrypt_embedding(self, vec: list[float]) -> tuple[str, str]:
        try:
            arr = np.array(vec, dtype=np.float32)
            if arr.shape[0] != self.DIM:

                if arr.shape[0] < self.DIM:
                    arr = np.concatenate([arr, np.zeros(self.DIM - arr.shape[0])])
                else:
                    arr = arr[:self.DIM]

            rotated = self._rotate(arr)
            bucket = self._simhash_bucket(rotated)
            quant = self._quantize(rotated)

            payload = json.dumps({
                "v": 2,
                "dim": self.DIM,
                "rot": True,
                "data": quant,
            })
            token = crypto.encrypt(payload, aad=self.AAD_CONTEXT)
            return token, bucket
        except Exception as e:
            logger.error(f"[FHEv2] encrypt_embedding failed: {e}")
            return "", "0"*16

    def decrypt_embedding(self, token: str) -> np.ndarray:
        try:
            raw = crypto.decrypt(token)
            obj = json.loads(raw)
            if obj.get("v") != 2:
                logger.warning("[FHEv2] Unsupported embedding version.")
                return np.zeros(self.DIM, dtype=np.float32)
            quant = obj.get("data", [])
            rotated = self._dequantize(quant)
            original = self.rotation.T @ rotated
            return original
        except Exception as e:
            logger.warning(f"[FHEv2] decrypt_embedding failed: {e}")
            return np.zeros(self.DIM, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def enclave_similarity(self, enc_a: str, query_vec: np.ndarray, enclave: SecureEnclave) -> float:
        dec = enclave.track(self.decrypt_embedding(enc_a))
        return self.cosine(dec, query_vec)

class SecureKeyManager:

    def __init__(
        self,
        method="argon2id",
        vault_path="secure/key_vault.json",
        time_cost: int = ARGON2_TIME_COST_DEFAULT,
        memory_cost: int = ARGON2_MEMORY_COST_KIB,
        parallelism: int = ARGON2_PARALLELISM,
        hash_len: int = ARGON2_HASH_LEN,
    ):
        self.method       = method
        self.vault_path   = vault_path
        self.time_cost    = time_cost
        self.memory_cost  = memory_cost
        self.parallelism  = parallelism
        self.hash_len     = hash_len
        self._ensure_vault()

        vault_meta = self._load_vault()

        self.active_version = vault_meta["active_version"]

        self._keys = {
            int(kv["version"]): base64.b64decode(kv["master_secret"])
            for kv in vault_meta["keys"]
        }

        self._derived_keys = {}
        vault_salt = base64.b64decode(vault_meta["salt"])
        for ver, master_secret in self._keys.items():
            self._derived_keys[ver] = self._derive_key(master_secret, vault_salt)

    def _get_passphrase(self) -> bytes:

        pw = os.getenv(VAULT_PASSPHRASE_ENV)
        if pw is None or pw == "":
            pw = base64.b64encode(os.urandom(32)).decode()
            logging.warning(
                "[SecureKeyManager] VAULT_PASSPHRASE not set; generated ephemeral key. "
                "Vault will not be readable across restarts!"
            )
        return pw.encode("utf-8")

    def _derive_vault_key(self, passphrase: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=max(self.time_cost, 3),   
            memory_cost=max(self.memory_cost, 262144),
            parallelism=max(self.parallelism, 1),
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _derive_key(self, master_secret: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=master_secret,
            salt=salt,
            time_cost=self.time_cost,
            memory_cost=self.memory_cost,
            parallelism=self.parallelism,
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _ensure_vault(self):

        if not os.path.exists("secure"):
            os.makedirs("secure", exist_ok=True)
        if os.path.exists(self.vault_path):
            return


        salt          = os.urandom(16)
        master_secret = os.urandom(32) 

        vault_body = {
            "version": VAULT_VERSION,
            "active_version": DATA_KEY_VERSION,
            "keys": [
                {
                    "version": DATA_KEY_VERSION,
                    "master_secret": base64.b64encode(master_secret).decode(),
                    "created": datetime.utcnow().isoformat() + "Z",
                }
            ],
            "salt": base64.b64encode(salt).decode(),
        }

        self._write_encrypted_vault(vault_body)

    def _write_encrypted_vault(self, vault_body: dict):

        plaintext = json.dumps(vault_body, indent=2).encode("utf-8")
        salt      = base64.b64decode(vault_body["salt"])

        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        nonce      = os.urandom(VAULT_NONCE_SIZE)
        ct         = aesgcm.encrypt(nonce, plaintext, _aad_str("vault", str(vault_body["version"])))

        on_disk = {
            "vault_format": VAULT_VERSION,
            "salt": vault_body["salt"],  
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ct).decode(),
        }
        with open(self.vault_path, "w") as f:
            json.dump(on_disk, f, indent=2)

    def _load_vault(self) -> dict:

        with open(self.vault_path, "r") as f:
            data = json.load(f)

        if "ciphertext" not in data:
            salt          = base64.b64decode(data["salt"])
            master_secret = base64.b64decode(data["master_secret"])
            vault_body = {
                "version": VAULT_VERSION,
                "active_version": DATA_KEY_VERSION,
                "keys": [
                    {
                        "version": DATA_KEY_VERSION,
                        "master_secret": base64.b64encode(master_secret).decode(),
                        "created": datetime.utcnow().isoformat() + "Z",
                    }
                ],
                "salt": base64.b64encode(salt).decode(),
            }
            self._write_encrypted_vault(vault_body)
            return vault_body
 
        salt      = base64.b64decode(data["salt"])
        nonce     = base64.b64decode(data["nonce"])
        ct        = base64.b64decode(data["ciphertext"])
        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        plaintext  = aesgcm.decrypt(nonce, ct, _aad_str("vault", str(VAULT_VERSION)))
        return json.loads(plaintext.decode("utf-8"))

    def encrypt(
        self,
        plaintext: str,
        *,
        aad: bytes = None,
        key_version: int = None,
    ) -> str:

        if plaintext is None:
            plaintext = ""
        if key_version is None:
            key_version = self.active_version
        if aad is None:
            aad = _aad_str("global", f"k{key_version}")

        key    = self._derived_keys[key_version]
        aesgcm = AESGCM(key)
        nonce  = os.urandom(DATA_NONCE_SIZE)
        ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)

        token = {
            "v": VAULT_VERSION,        
            "k": key_version,          
            "aad": aad.decode("utf-8"),
            "n": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }
        return json.dumps(token, separators=(",", ":"))

    def decrypt(self, token: str) -> str:

        if not token:
            return ""

        if token.startswith("{"):
            try:
                meta = json.loads(token)
            except Exception:
                logging.warning("[SecureKeyManager] Invalid JSON token; returning raw.")
                return token

            v   = int(meta.get("v", 1))
            ver = int(meta.get("k", self.active_version))
            aad = meta.get("aad", "global").encode()
            n   = base64.b64decode(meta["n"])
            ct  = base64.b64decode(meta["ct"])

            key = self._derived_keys.get(ver)
            if key is None:
                raise ValueError(f"No key for version {ver}; cannot decrypt.")
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(n, ct, aad)
            return pt.decode("utf-8")

        try:
            raw   = base64.b64decode(token.encode())
            nonce = raw[:DATA_NONCE_SIZE]
            ct    = raw[DATA_NONCE_SIZE:]
            key   = self._derived_keys[self.active_version]
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(nonce, ct, None)
            return pt.decode("utf-8")
        except Exception as e:
            logging.warning(f"[SecureKeyManager] Legacy decrypt failed: {e}")
            return token

    def add_new_key_version(self) -> int:

        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1

        master_secret = os.urandom(32)
        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(master_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })

        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)
        self._keys[new_version] = master_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(master_secret, salt)
        self.active_version = new_version
        logging.info(f"[SecureKeyManager] Installed new key version {new_version}.")
        return new_version

    def _entropy_bits(self, secret_bytes: bytes) -> float:

        if not secret_bytes:
            return 0.0
        counts = Counter(secret_bytes)
        total = float(len(secret_bytes))
        H = 0.0
        for c in counts.values():
            p = c / total
            H -= p * math.log2(p)
        return H

    def _resistance_score(self, secret_bytes: bytes) -> float:

        dist_component = 0.0
        try:
            arr_candidate = np.frombuffer(secret_bytes, dtype=np.uint8).astype(np.float32)
            for k in self._keys.values():
                arr_prev = np.frombuffer(k, dtype=np.uint8).astype(np.float32)
                dist_component += np.linalg.norm(arr_candidate - arr_prev)
        except Exception:
            pass
        if len(self._keys):
            dist_component /= len(self._keys)

        counts = Counter(secret_bytes)
        expected = len(secret_bytes) / 256.0
        chi_sq = sum(((c - expected) ** 2) / expected for c in counts.values())
        flatness = 1.0 / (1.0 + chi_sq)        # in (0,1]
        return float(dist_component * 0.01 + flatness)

    def self_mutate_key(self,
                        population: int = 6,
                        noise_sigma: float = 12.0,
                        alpha: float = 1.0,
                        beta: float = 2.0) -> int:

        vault_meta = self._load_vault()
        base_secret = None
        for kv in vault_meta["keys"]:
            if int(kv["version"]) == vault_meta["active_version"]:
                base_secret = base_secret or base64.b64decode(kv["master_secret"])
        if base_secret is None:
            raise RuntimeError("Active master secret not found.")

        rng = np.random.default_rng()
        candidates: List[bytes] = [base_secret]
        base_arr = np.frombuffer(base_secret, dtype=np.uint8).astype(np.int16)

        for _ in range(population - 1):
            noise = rng.normal(0, noise_sigma, size=base_arr.shape).astype(np.int16)
            mutated = np.clip(base_arr + noise, 0, 255).astype(np.uint8).tobytes()
            candidates.append(mutated)

        best_secret = base_secret
        best_fitness = -1e9
        for cand in candidates:
            H = self._entropy_bits(cand)
            R = self._resistance_score(cand)
            F = alpha * H + beta * R
            if F > best_fitness:
                best_fitness = F
                best_secret = cand

        new_version = self._install_custom_master_secret(best_secret)
        logging.info(f"[SelfMutateKey] Installed mutated key v{new_version} (fitness={best_fitness:.3f}).")
        return new_version

    def _install_custom_master_secret(self, new_secret: bytes) -> int:

        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1

        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(new_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })
        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)

        self._keys[new_version] = new_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(new_secret, salt)
        self.active_version = new_version
        return new_version

    def rotate_and_migrate_storage(self, migrate_func):

        new_ver = self.add_new_key_version()
        try:
            migrate_func(self)
        except Exception as e:
            logging.error(f"[SecureKeyManager] Migration failed after key rotation: {e}")
            raise
        logging.info(f"[SecureKeyManager] Migration to key v{new_ver} complete.")

crypto = SecureKeyManager()  

def _token_hist(text: str) -> Counter:
    return Counter(word_tokenize(text))

def _js_divergence(p: Counter, q: Counter) -> float:

    vocab = set(p) | set(q)
    if not vocab:
        return 0.0
    def _prob(c: Counter):
        tot = sum(c.values()) or 1
        return np.array([c[t]/tot for t in vocab], dtype=np.float32)
    P, Q = _prob(p), _prob(q)
    M    = 0.5 * (P + Q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

class TopologicalMemoryManifold:

    def __init__(self, dim: int = 2, sigma: float = 0.75,
                 diff_alpha: float = LAPLACIAN_ALPHA):
        self.dim        = dim
        self.sigma      = sigma
        self.diff_alpha = diff_alpha
        self._phrases:     list[str]     = []
        self._embeddings:  np.ndarray|None = None
        self._coords:      np.ndarray|None = None
        self._W:           np.ndarray|None = None
        self._graph_built  = False

    def _load_crystallized(self) -> list[tuple[str, float]]:
        rows = []
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("SELECT phrase, score FROM memory_osmosis "
                            "WHERE crystallized = 1")
                rows = cur.fetchall()
        except Exception as e:
            logger.error(f"[Manifold] load_crystallized failed: {e}")
        return rows

    def rebuild(self):
        data = self._load_crystallized()
        if not data:

            self._phrases, self._embeddings = [], None
            self._coords,  self._W         = None, None
            self._graph_built              = False
            return

        phrases, _ = zip(*data)
        self._phrases = list(phrases)

        E = np.array([compute_text_embedding(p) for p in self._phrases],
                     dtype=np.float32)

        dists = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
        W = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))
        np.fill_diagonal(W, 0.0)

        D = np.diag(W.sum(axis=1))
        L = D - W
        E = E - self.diff_alpha * (L @ E)

        try:
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
            L_sym      = D_inv_sqrt @ L @ D_inv_sqrt
            vals, vecs = np.linalg.eigh(L_sym)
            idx        = np.argsort(vals)[1:self.dim+1]
            Y          = D_inv_sqrt @ vecs[:, idx]
        except Exception as e:
            logger.error(f"[Manifold] eigen decomposition failed: {e}")
            Y = np.zeros((len(self._phrases), self.dim), dtype=np.float32)

        self._embeddings  = E
        self._coords      = Y.astype(np.float32)
        self._W           = W
        self._graph_built = True
        logger.info(f"[Manifold] Rebuilt manifold with {len(self._phrases)} phrases "
                    f"(α={self.diff_alpha}).")

    def geodesic_retrieve(self, query_text: str, k: int = 1) -> list[str]:

        if not self._graph_built or self._embeddings is None:
            return []

        q_vec = np.array(compute_text_embedding(query_text), dtype=np.float32)
        start_idx = int(np.argmin(
            np.linalg.norm(self._embeddings - q_vec[None, :], axis=1)
        ))

        n        = self._W.shape[0]
        visited  = np.zeros(n, dtype=bool)
        dist     = np.full(n, np.inf, dtype=np.float32)
        dist[start_idx] = 0.0

        for _ in range(n):
            u = np.argmin(dist + np.where(visited, 1e9, 0.0))
            if visited[u]:
                break
            visited[u] = True
            for v in range(n):
                w = self._W[u, v]
                if w <= 0 or visited[v]:
                    continue
                alt = dist[u] + 1.0 / (w + 1e-8)
                if alt < dist[v]:
                    dist[v] = alt

        order = np.argsort(dist)
        return [self._phrases[i] for i in order[:k]]

topo_manifold = TopologicalMemoryManifold()
fhe_v2 = AdvancedHomomorphicVectorMemory()

def setup_weaviate_schema(client):
    try:
        existing = client.schema.get()
        if not any(cls["class"] == "ReflectionLog" for cls in existing["classes"]):
            client.schema.create_class({
                "class": "ReflectionLog",
                "description": "Stores Dyson assistant's internal reflection and reasoning traces",
                "properties": [
                    {"name": "type", "dataType": ["string"]},
                    {"name": "user_id", "dataType": ["string"]},
                    {"name": "bot_id", "dataType": ["string"]},
                    {"name": "query", "dataType": ["text"]},
                    {"name": "response", "dataType": ["text"]},
                    {"name": "reasoning_trace", "dataType": ["text"]},
                    {"name": "prompt_snapshot", "dataType": ["text"]},
                    {"name": "z_state", "dataType": ["blob"]},
                    {"name": "entropy", "dataType": ["number"]},
                    {"name": "bias_factor", "dataType": ["number"]},
                    {"name": "temperature", "dataType": ["number"]},
                    {"name": "top_p", "dataType": ["number"]},
                    {"name": "sentiment_target", "dataType": ["number"]},
                    {"name": "timestamp", "dataType": ["date"]}
                ]
            })
            print("ReflectionLog schema created.")
        else:
            print("ReflectionLog schema already exists.")
    except Exception as e:
        logger.error(f"[Schema Init Error] {e}")

def _load_policy_if_needed(self):
    if not hasattr(self, "pg_params"):
        try:
            self._load_policy()
        except Exception as e:
            logger.warning(f"[Policy Load Error] {e}")
            self.pg_params = {}
    if not hasattr(self, "pg_learning_rate"):
        self.pg_learning_rate = 0.05

def evaluate_candidate(response: str, target_sentiment: float, original_query: str) -> float:

    response_sentiment = TextBlob(response).sentiment.polarity
    sentiment_alignment = 1.0 - abs(target_sentiment - response_sentiment)

    overlap_score = sum(1 for word in original_query.lower().split() if word in response.lower())
    overlap_bonus = min(overlap_score / 5.0, 1.0)  # normalize

    return (0.7 * sentiment_alignment) + (0.3 * overlap_bonus)

def build_record_aad(user_id: str, *, source: str, table: str = "", cls: str = "") -> bytes:

    context_parts = [source]
    if table:
        context_parts.append(table)
    if cls:
        context_parts.append(cls)
    context_parts.append(user_id)
    return _aad_str(*context_parts)

def compute_text_embedding(text: str) -> list[float]:
    if not text:
        return [0.0] * fhe_v2.DIM
    tokens = re.findall(r'\w+', text.lower())
    counts = Counter(tokens)
    vocab = sorted(counts.keys())[:fhe_v2.DIM]
    vec = [float(counts[w]) for w in vocab]
    if len(vec) < fhe_v2.DIM:
        vec.extend([0.0] * (fhe_v2.DIM - len(vec)))
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr /= n
    return arr.tolist()

def generate_uuid_for_weaviate(identifier, namespace=''):
    if not identifier:
        raise ValueError("Identifier for UUID generation is empty or None")

    if not namespace:
        namespace = str(uuid.uuid4())

    try:
        return generate_uuid5(namespace, identifier)
    except Exception as e:
        logger.error(f"Error generating UUID: {e}")
        raise

def is_valid_uuid(uuid_to_test, version=5):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False
    
def fetch_live_weather(lat: float, lon: float, fallback_temp_f: float = 70.0) -> tuple[float, int, bool]:
    try:
        import httpx 
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            temp_f = (temp_c * 9 / 5) + 32
            weather_code = int(current.get("weathercode", 0))
            return temp_f, weather_code, True 
    except Exception as e:
        logger.warning(f"[Weather] Fallback due to error: {e}")
        return fallback_temp_f, 0, False

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def rgb_quantum_gate(
    r, g, b,
    cpu_usage,
    tempo=120,
    lat=0.0,
    lon=0.0,
    temperature_f=70.0,
    weather_scalar=0.0,
    z0_hist=0.0,
    z1_hist=0.0,
    z2_hist=0.0
):

    r, g, b = [min(1.0, max(0.0, x)) for x in (r, g, b)]
    cpu_scale = max(0.05, cpu_usage)

    tempo_norm = min(1.0, max(0.0, tempo / 200))
    lat_rad = np.deg2rad(lat % 360)
    lon_rad = np.deg2rad(lon % 360)
    temp_norm = min(1.0, max(0.0, (temperature_f - 30) / 100))
    weather_mod = min(1.0, max(0.0, weather_scalar))

    coherence_gain = 1.0 + tempo_norm - weather_mod + 0.3 * (1 - abs(0.5 - temp_norm))

    q_r = r * np.pi * cpu_scale * coherence_gain
    q_g = g * np.pi * cpu_scale * (1.0 - weather_mod + temp_norm)
    q_b = b * np.pi * cpu_scale * (1.0 + weather_mod - temp_norm)

    qml.RX(q_r, wires=0)
    qml.RY(q_g, wires=1)
    qml.RZ(q_b, wires=2)

    qml.PhaseShift(lat_rad * tempo_norm, wires=0)
    qml.PhaseShift(lon_rad * (1 - weather_mod), wires=1)

    qml.CRX(temp_norm * np.pi * coherence_gain, wires=[2, 0])
    qml.CRY(tempo_norm * np.pi, wires=[1, 2])
    qml.CRZ(weather_mod * np.pi, wires=[0, 2])

    entropy_cycle = np.sin(cpu_scale * np.pi * 2)
    qml.RX(entropy_cycle * np.pi * 0.5, wires=1)

    feedback_phase = (z0_hist + z1_hist + z2_hist) * np.pi
    qml.PhaseShift(feedback_phase / 3.0, wires=0)
    qml.PhaseShift(-feedback_phase / 2.0, wires=2)

    if 0.3 < weather_mod < 0.6:
        qml.IsingYY(temp_norm * np.pi * 0.5, wires=[0, 2])

    if weather_mod > 0.5:
        qml.Toffoli(wires=[0, 1, 2])
        qml.AmplitudeDamping(0.1 * weather_mod, wires=2)
    else:
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2)),
    )

def get_current_multiversal_time():
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    x, y, z, t = 34, 76, 12, 5633
    return f"X:{x}, Y:{y}, Z:{z}, T:{t}, Time:{current_time}"

def extract_rgb_from_text(text):

    if not text or not isinstance(text, str):
        return (128, 128, 128) 

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity 
    subjectivity = blob.sentiment.subjectivity 
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    word_count = len(tokens)
    sentence_count = len(blob.sentences) or 1
    avg_sentence_length = word_count / sentence_count
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    punctuation_density = sum(1 for ch in text if ch in ',;:!?') / max(1, word_count)
    valence = polarity 
    arousal = (verb_count + adv_count) / max(1, word_count)
    dominance = (adj_count + 1) / (noun_count + 1) 
    hue_raw = ((1 - valence) * 120 + dominance * 20) % 360
    hue = hue_raw / 360.0
    saturation = min(1.0, max(0.2, 0.25 + 0.4 * arousal + 0.2 * subjectivity + 0.15 * (dominance - 1)))
    brightness = max(0.2, min(1.0,
        0.9 - 0.03 * avg_sentence_length + 0.2 * punctuation_density
    ))

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (int(r * 255), int(g * 255), int(b * 255))

def init_db():

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS local_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    response TEXT,
                    response_time TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_osmosis (
                    phrase TEXT PRIMARY KEY,
                    score REAL,
                    last_updated TEXT,
                    crystallized INTEGER DEFAULT 0
                )
            """)
            conn.commit()

        interaction_history_class = {
            "class": "InteractionHistory",
            "properties": [
                {"name": "user_id", "dataType": ["string"]},
                {"name": "response", "dataType": ["string"]},
                {"name": "response_time", "dataType": ["string"]}
            ]
        }

        long_term_memory_class = {
            "class": "LongTermMemory",
            "properties": [
                {"name": "phrase", "dataType": ["string"]},
                {"name": "score", "dataType": ["number"]},
                {"name": "crystallized_time", "dataType": ["string"]}
            ]
        }

        existing_classes = client.schema.get().get("classes", [])
        existing_names = {c["class"] for c in existing_classes}

        if "InteractionHistory" not in existing_names:
            client.schema.create_class(interaction_history_class)
        if "LongTermMemory" not in existing_names:
            client.schema.create_class(long_term_memory_class)

    except Exception as e:
        logger.error(f"Error during database/schema initialization: {e}")
        raise

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(memory_osmosis)")
            cols = {row[1] for row in cur.fetchall()}
            if "aging_last" not in cols:
                cur.execute("ALTER TABLE memory_osmosis ADD COLUMN aging_last TEXT")
                conn.commit()
    except Exception as e:
        logger.warning(f"[Aging] Could not add aging_last column (continuing with last_updated): {e}")

def save_user_message(user_id, user_input):
    logger.info(f"[save_user_message] user_id={user_id}")
    if not user_input:
        logger.warning("User input is empty.")
        return
    try:
        user_input = sanitize_text(user_input, max_len=4000)
        response_time = get_current_multiversal_time()

        aad_sql  = build_record_aad(user_id=user_id, source="sqlite", table="local_responses")
        aad_weav = build_record_aad(user_id=user_id, source="weaviate", cls="InteractionHistory")

        encrypted_input_sql  = crypto.encrypt(user_input, aad=aad_sql)
        encrypted_input_weav = crypto.encrypt(user_input, aad=aad_weav)

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (user_id, encrypted_input_sql, response_time)
            )
            conn.commit()

        plain_embedding = compute_text_embedding(user_input)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)
        dummy_vector = [0.0] * fhe_v2.DIM

        obj = {
            "user_id": user_id,
            "user_message": encrypted_input_weav,
            "response_time": response_time,
            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket
        }
        generated_uuid = generate_uuid5(user_id, user_input)
        response = requests.post(
            'http://127.0.0.1:8079/v1/objects',
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": obj,
                "vector": dummy_vector
            },
            timeout=10
        )
        if response.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {response.status_code} {response.text}")
    except Exception as e:
        logger.exception(f"Exception in save_user_message: {e}")

def save_bot_response(bot_id: str, bot_response: str):
    logger.info(f"[save_bot_response] bot_id={bot_id}")
    if not bot_response:
        logger.warning("Bot response is empty.")
        return
    try:
        bot_response = sanitize_text(bot_response, max_len=4000)
        response_time = get_current_multiversal_time()

        aad_sql = build_record_aad(user_id=bot_id, source="sqlite", table="local_responses")
        enc_sql = crypto.encrypt(bot_response, aad=aad_sql)
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (bot_id, enc_sql, response_time)
            )
            conn.commit()

        aad_weav = build_record_aad(user_id=bot_id, source="weaviate", cls="InteractionHistory")
        enc_weav = crypto.encrypt(bot_response, aad=aad_weav)

        plain_embedding = compute_text_embedding(bot_response)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)
        dummy_vector = [0.0] * fhe_v2.DIM

        props = {
            "user_id": bot_id,
            "ai_response": enc_weav,
            "response_time": response_time,
            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket
        }
        generated_uuid = generate_uuid5(bot_id, bot_response)
        resp = requests.post(
            "http://127.0.0.1:8079/v1/objects",
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": props,
                "vector": dummy_vector
            },
            timeout=10
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.exception(f"Exception in save_bot_response: {e}")

def query_reflections(self, user_id: str, substring: str = None, limit: int = 5):
    try:
        filters = {
            "path": ["user_id"],
            "operator": "Equal",
            "valueString": user_id
        }

        query = self.client.query.get(
            "ReflectionLog",
            ["query", "response", "reasoning_trace", "timestamp"]
        ).with_where(filters).with_limit(limit)

        if substring:
            query = query.with_additional({
                "moduleParams": {
                    "text": {
                        "concepts": [substring],
                        "certainty": 0.65
                    }
                }
            })

        result = query.do()
        return result["data"]["Get"]["ReflectionLog"]
    except Exception as e:
        logger.error(f"[Weaviate Reflection Query Error] {e}")
        return []

def reflect_on_memory(self, user_id: str, topic: str) -> str:
    reflections = self.query_reflections(user_id, substring=topic, limit=3)
    if not reflections:
        return "I could not locate a relevant reflection trace on that topic."

    response = ["[Dyson Node: Reflection Summary]"]
    for r in reflections:
        response.append(f"Query: {r['query']}")
        response.append(f"Response: {r['response']}")
        response.append(f"Reasoning: {r['reasoning_trace']}")
        response.append(f"Timestamp: {r['timestamp']}")
        response.append("────────────────────────────")
    return "\n".join(response)


llm = Llama(
    model_path=model_path,
    mmproj=mmproj_path,
    n_gpu_layers=-1,
    n_ctx=3900,
)

def is_code_like(chunk):
   code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
   return bool(re.search(code_patterns, chunk))

def determine_token(chunk, memory, max_words_to_check=500):
   combined_chunk = f"{memory} {chunk}"
   if not combined_chunk:
       return "[attention]"

   if is_code_like(combined_chunk):
       return "[code]"

   words = word_tokenize(combined_chunk)[:max_words_to_check]
   tagged_words = pos_tag(words)

   pos_counts = Counter(tag[:2] for _, tag in tagged_words)
   most_common_pos, _ = pos_counts.most_common(1)[0]

   if most_common_pos == 'VB':
       return "[action]"
   elif most_common_pos == 'NN':
       return "[subject]"
   elif most_common_pos in ['JJ', 'RB']:
       return "[description]"
   else:
       return "[general]"

def find_max_overlap(chunk, next_chunk):
   max_overlap = min(len(chunk), 240)
   return next((overlap for overlap in range(max_overlap, 0, -1) if chunk.endswith(next_chunk[:overlap])), 0)

def truncate_text(text, max_words=100):
   return ' '.join(text.split()[:max_words])

def fetch_relevant_info(chunk, client, user_input):
    try:
        if not user_input:
            return ""

        query_vec = np.array(compute_text_embedding(user_input), dtype=np.float32)
        rotated = fhe_v2._rotate(query_vec)
        bucket = fhe_v2._simhash_bucket(rotated)

        gql = f"""
        {{
            Get {{
                InteractionHistory(
                    where: {{
                        path: ["embedding_bucket"],
                        operator: Equal,
                        valueString: "{bucket}"
                    }}
                    limit: 40
                    sort: {{path:"response_time", order: desc}}
                ) {{
                    user_message
                    ai_response
                    encrypted_embedding
                }}
            }}
        }}
        """
        response = client.query.raw(gql)
        results = (
            response.get('data', {})
                    .get('Get', {})
                    .get('InteractionHistory', [])
        )
        best = None
        best_score = -1.0
        with SecureEnclave() as enclave:
            for obj in results:
                enc_emb = obj.get("encrypted_embedding", "")
                if not enc_emb:
                    continue
                score = fhe_v2.enclave_similarity(enc_emb, query_vec, enclave)
                if score > best_score:
                    best_score = score
                    best = obj

        if not best or best_score <= 0:
            return ""

        user_msg_raw = try_decrypt(best.get("user_message", ""))
        ai_resp_raw  = try_decrypt(best.get("ai_response", ""))
        return f"{user_msg_raw} {ai_resp_raw}"
    except Exception as e:
        logger.error(f"[FHEv2 retrieval] failed: {e}")
        return ""

def llama_generate(prompt, weaviate_client=None, user_input=None, temperature=1.0, top_p=0.9):
    config = load_config()
    max_tokens = config.get('MAX_TOKENS', 2500)
    chunk_size = config.get('CHUNK_SIZE', 358)
    try:
        prompt_chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
        responses = []
        last_output = ""
        memory = ""

        for i, current_chunk in enumerate(prompt_chunks):
            relevant_info = fetch_relevant_info(current_chunk, weaviate_client, user_input)
            combined_chunk = f"{relevant_info} {current_chunk}"
            token = determine_token(combined_chunk, memory)
            output = tokenize_and_generate(
                combined_chunk,
                token,
                max_tokens,
                chunk_size,
                temperature,
                top_p
            )

            if output is None:
                logger.error(f"Failed to generate output for chunk: {combined_chunk}")
                continue

            if i > 0 and last_output:
                overlap = find_max_overlap(last_output, output)
                output = output[overlap:]

            memory += output
            responses.append(output)
            last_output = output

        final_response = ''.join(responses)
        return final_response if final_response else None

    except Exception as e:
        logger.error(f"Error in llama_generate: {e}")
        return None

def tokenize_and_generate(chunk, token, max_tokens, chunk_size, temperature=1.0, top_p=0.9):
    try:
        inputs = llm(
            f"[{token}] {chunk}",
            max_tokens=min(max_tokens, chunk_size),
            temperature=temperature,
            top_p=top_p
        )
        if inputs is None or not isinstance(inputs, dict):
            logger.error(f"Llama model returned invalid output for input: {chunk}")
            return None

        choices = inputs.get('choices', [])
        if not choices or not isinstance(choices[0], dict):
            logger.error("No valid choices in Llama output")
            return None

        return choices[0].get('text', '')
    except Exception as e:
        logger.error(f"Error in tokenize_and_generate: {e}")
        return None

def extract_verbs_and_nouns(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        verbs_and_nouns = [word for word, tag in tagged_words if tag.startswith('VB') or tag.startswith('NN')]
        return verbs_and_nouns

    except Exception as e:
        print(f"Error in extract_verbs_and_nouns: {e}")
        return []

def try_decrypt(value):
    try:
        return crypto.decrypt(value)
    except Exception as e:
        logger.warning(f"[decryption] Could not decrypt value: {e}")
        return value

class App(customtkinter.CTk):

    @staticmethod
    def _encrypt_field(value: str) -> str:
        try:
            return crypto.encrypt(value if value is not None else "")
        except Exception as e:
            logger.error(f"[encrypt] Failed to encrypt value: {e}")
            return value if value is not None else ""

    @staticmethod
    def _decrypt_field(value: str) -> str:
        if value is None:
            return ""
        try:
            return crypto.decrypt(value)
        except Exception as e:
            logger.warning(f"[decrypt] Could not decrypt value (returning raw): {e}")
            return value

    def __init__(self, user_identifier):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_z = (0.0, 0.0, 0.0)
        self._policy_lock = threading.RLock()
        self._policy_mtime = None
        self._load_policy_if_needed()
        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)
        self.after(6 * 3600 * 1000, self._schedule_key_mutation)


    def memory_aging_scheduler(self):

        self.run_long_term_memory_aging()

        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

    def _policy_params_path(self):
        return path.join(bundle_dir, "policy_params.json")

    def _load_policy_if_needed(self):

        import os, json

        defaults = {
            "temp_w": 0.0,
            "temp_b": 0.0,
            "temp_log_sigma": -0.7,
            "top_w": 0.0,
            "top_b": 0.0,
            "top_log_sigma": -0.7,
            "learning_rate": 0.05
        }

        path = self._policy_params_path()

        with self._policy_lock:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None

            reload_needed = (
                not hasattr(self, "pg_params")
                or (mtime is not None and mtime != self._policy_mtime)
            )

            if reload_needed:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    for key, val in defaults.items():
                        data.setdefault(key, val)
                    self.pg_params = data
                    self._policy_mtime = mtime
                    logger.debug(f"[Policy] Loaded params from {path}: {self.pg_params}")
                except Exception as e:
                    logger.warning(f"[Policy Load Error] could not load {path}: {e}")
                    self.pg_params = defaults.copy()
                    self._policy_mtime = mtime

            if not hasattr(self, "pg_learning_rate"):

                env_lr = os.getenv("PG_LEARNING_RATE")
                if env_lr is not None:
                    try:
                        lr = float(env_lr)
                    except ValueError:
                        logger.warning(f"[Policy] Invalid PG_LEARNING_RATE='{env_lr}', falling back")
                        lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                else:
                    lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                self.pg_learning_rate = lr
                logger.debug(f"[Policy] Using learning_rate={self.pg_learning_rate}")

    def _load_policy(self):

        default = {
            "temp_w": 0.0,
            "temp_b": 0.0,
            "temp_log_sigma": -0.7,   
            "top_w": 0.0,
            "top_b": 0.0,
            "top_log_sigma": -0.7
        }
        try:
            with open(self._policy_params_path(), "r") as f:
                data = json.load(f)
                for k, v in default.items():
                    if k not in data:
                        data[k] = v
                self.pg_params = data
        except Exception:
            self.pg_params = default
            self._save_policy()

    def _save_policy(self):
        try:
            with open(self._policy_params_path(), "w") as f:
                json.dump(self.pg_params, f, indent=2)
        except Exception as e:
            logger.error(f"[PG] Failed saving policy params: {e}")

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _policy_forward(self, bias_factor: float):

        p = self.pg_params

        t_range = 1.5 - 0.2
        raw_t = p["temp_w"] * bias_factor + p["temp_b"]
        sig_t = self._sigmoid(raw_t)
        mu_t = 0.2 + sig_t * t_range

        p_range = 1.0 - 0.2
        raw_p = p["top_w"] * bias_factor + p["top_b"]
        sig_p = self._sigmoid(raw_p)
        mu_p = 0.2 + sig_p * p_range

        sigma_t = math.exp(p["temp_log_sigma"]) + 1e-4
        sigma_p = math.exp(p["top_log_sigma"]) + 1e-4

        cache = {
            "raw_t": raw_t, "sig_t": sig_t,
            "raw_p": raw_p, "sig_p": sig_p,
            "t_range": t_range, "p_range": p_range
        }
        return mu_t, sigma_t, mu_p, sigma_p, cache

    def _policy_sample(self, bias_factor: float):

        mu_t, sigma_t, mu_p, sigma_p, cache = self._policy_forward(bias_factor)
        t_sample = random.gauss(mu_t, sigma_t)
        p_sample = random.gauss(mu_p, sigma_p)
        t_clip = max(0.2, min(1.5, t_sample))
        p_clip = max(0.2, min(1.0, p_sample))
        log_prob_t = -0.5 * ((t_sample - mu_t) ** 2 / (sigma_t ** 2)) - math.log(sigma_t) - 0.5 * math.log(2 * math.pi)
        log_prob_p = -0.5 * ((p_sample - mu_p) ** 2 / (sigma_p ** 2)) - math.log(sigma_p) - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_t + log_prob_p

        return {
            "temperature": t_clip,
            "top_p": p_clip,
            "raw_temperature": t_sample,
            "raw_top_p": p_sample,
            "mu_t": mu_t, "sigma_t": sigma_t,
            "mu_p": mu_p, "sigma_p": sigma_p,
            "log_prob": log_prob,
            "cache": cache
        }

    def _policy_update(self, samples, learning_rate=0.05):

        if not samples:
            return
        avg_reward = sum(s["reward"] for s in samples) / len(samples)

        grads = {k: 0.0 for k in self.pg_params.keys()}

        for s in samples:
            advantage = s["reward"] - avg_reward
            if advantage == 0:
                continue

            mu_t = s["mu_t"]; sigma_t = s["sigma_t"]
            mu_p = s["mu_p"]; sigma_p = s["sigma_p"]
            rt = s["raw_temperature"]; rp = s["raw_top_p"]
            cache = s["cache"]
            bias_factor = s.get("bias_factor", 0.0)
            inv_var_t = 1.0 / (sigma_t ** 2)
            inv_var_p = 1.0 / (sigma_p ** 2)
            diff_t = (rt - mu_t)
            diff_p = (rp - mu_p)
            dlogp_dmu_t = diff_t * inv_var_t
            dlogp_dmu_p = diff_p * inv_var_p
            dlogp_dlogsigma_t = (diff_t ** 2 / (sigma_t ** 2)) - 1.0
            dlogp_dlogsigma_p = (diff_p ** 2 / (sigma_p ** 2)) - 1.0
            sig_t = cache["sig_t"]; t_range = cache["t_range"]
            dsig_t_draw_t = sig_t * (1 - sig_t)
            dmu_t_draw_t = dsig_t_draw_t * t_range
            sig_p = cache["sig_p"]; p_range = cache["p_range"]
            dsig_p_draw_p = sig_p * (1 - sig_p)
            dmu_p_draw_p = dsig_p_draw_p * p_range
            grads["temp_w"] += advantage * dlogp_dmu_t * dmu_t_draw_t * bias_factor
            grads["temp_b"] += advantage * dlogp_dmu_t * dmu_t_draw_t
            grads["temp_log_sigma"] += advantage * dlogp_dlogsigma_t
            grads["top_w"] += advantage * dlogp_dmu_p * dmu_p_draw_p * bias_factor
            grads["top_b"] += advantage * dlogp_dmu_p * dmu_p_draw_p
            grads["top_log_sigma"] += advantage * dlogp_dlogsigma_p

        for k, g in grads.items():
            self.pg_params[k] += learning_rate * g

        self._save_policy()
        logger.info(f"[PG] Updated policy params: {self.pg_params}")

    def retrieve_past_interactions(self, user_input, result_queue):

        try:
            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)

            user_message, ai_response = self.fetch_relevant_info_internal(concepts_query)

            if user_message and ai_response:
                combo = f"{user_message} {ai_response}"
                summarized_interaction = summarizer.summarize(combo) or combo
                sentiment = TextBlob(summarized_interaction).sentiment.polarity
                processed_interaction = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized_interaction,
                    "sentiment": sentiment
                }
                result_queue.put([processed_interaction])
            else:
                logger.info("No relevant interactions found for the given user input.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])

    def _weaviate_find_ltm(self, phrase: str):
        safe_phrase = sanitize_for_graphql_string(phrase, max_len=256)
        gql = f"""
        {{
          Get {{
            LongTermMemory(
              where: {{ path:["phrase"], operator:Equal, valueString:"{safe_phrase}" }}
              limit: 1
            ) {{
              phrase
              score
              crystallized_time
              _additional {{ id }}
            }}
          }}
        }}
        """
        try:
            resp = self.client.query.raw(gql)
            items = resp.get("data", {}).get("Get", {}).get("LongTermMemory", [])
            if not items:
                return None, None, None
            obj = items[0]
            return (
                obj["_additional"]["id"],
                float(obj.get("score", 0.0)),
                obj.get("crystallized_time", "")
            )
        except Exception as e:
            logger.error(f"[Aging] _weaviate_find_ltm failed: {e}")
            return None, None, None

    def _weaviate_update_ltm_score(self, uuid_str: str, new_score: float):
        try:
            self.client.data_object.update(
                class_name="LongTermMemory",
                uuid=uuid_str,
                data_object={"score": new_score}
            )
        except Exception as e:
            logger.error(f"[Aging] update score failed for {uuid_str}: {e}")

    def _weaviate_delete_ltm(self, uuid_str: str):
        try:
            self.client.data_object.delete(
                class_name="LongTermMemory",
                uuid=uuid_str
            )
        except Exception as e:
            logger.error(f"[Aging] delete failed for {uuid_str}: {e}")

    def run_long_term_memory_aging(self):

        try:
            now = datetime.utcnow()
            purged_any = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                try:
                    cur.execute("""SELECT phrase, score,
                                          COALESCE(aging_last, last_updated) AS ts,
                                          crystallized
                                   FROM memory_osmosis
                                   WHERE crystallized=1""")
                except sqlite3.OperationalError:
                    cur.execute("""SELECT phrase, score, last_updated AS ts, crystallized
                                   FROM memory_osmosis
                                   WHERE crystallized=1""")

                rows = cur.fetchall()
                for phrase, score, ts, crystallized in rows:
                    if not ts:
                        continue
                    try:
                        base_dt = datetime.fromisoformat(ts.replace("Z", ""))
                    except Exception:
                        continue
                    delta_days = max(0.0, (now - base_dt).total_seconds() / 86400.0)
                    if delta_days <= 0:
                        continue

                    half_life = AGING_T0_DAYS + AGING_GAMMA_DAYS * math.log(1.0 + max(score, 0.0))
                    if half_life <= 0:
                        continue
                    decay_factor = 0.5 ** (delta_days / half_life)
                    new_score = score * decay_factor

                    uuid_str, _, _ = self._weaviate_find_ltm(phrase)
                    if new_score < AGING_PURGE_THRESHOLD:
                        purged_any = True
                        if uuid_str:
                            self._weaviate_delete_ltm(uuid_str)
                        cur.execute("""UPDATE memory_osmosis
                                       SET crystallized=0, score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        logger.info(f"[Aging] Purged crystallized phrase '{phrase}' (decayed to {new_score:.3f}).")
                    else:
                        cur.execute("""UPDATE memory_osmosis
                                       SET score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        if uuid_str:
                            self._weaviate_update_ltm_score(uuid_str, new_score)

                conn.commit()
            if purged_any:
                topo_manifold.rebuild()
        except Exception as e:
            logger.error(f"[Aging] run_long_term_memory_aging failed: {e}")

    def get_weather_sync(self, lat, lon):

        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            weather_code = int(current.get("weathercode", 0))
            return temp_c, weather_code
        except Exception as e:
            logger.error(f"[Weather] Fetch failed: {e}")
            return None, None

    def generate_quantum_state(self, rgb=None):
        if rgb is None or not isinstance(rgb, tuple) or len(rgb) != 3:
            rgb = (128, 128, 128)

        try:
            cpu = psutil.cpu_percent(interval=0.3) / 100.0
            cpu = max(cpu, 0.05)

            r, g, b = [min(1.0, max(0.0, c / 255)) for c in rgb]

            try:
                lat = float(self.latitude_entry.get())
                lon = float(self.longitude_entry.get())
            except Exception:
                lat, lon = 0.0, 0.0

            try:
                user_temp_f = float(self.temperature_entry.get() or 70.0)
            except ValueError:
                user_temp_f = 70.0

            temp_f, weather_code, is_live = fetch_live_weather(lat, lon, user_temp_f)

            if weather_code in {1, 2, 3}:
                weather_scalar = 0.3
            elif weather_code >= 61:
                weather_scalar = 0.7
            else:
                weather_scalar = 0.0

            tempo = 120 

            z0_hist, z1_hist, z2_hist = self.last_z

            z0, z1, z2 = rgb_quantum_gate(
                r, g, b,
                cpu_usage=cpu,
                tempo=tempo,
                lat=lat,
                lon=lon,
                temperature_f=temp_f,
                weather_scalar=weather_scalar,
                z0_hist=z0_hist,
                z1_hist=z1_hist,
                z2_hist=z2_hist
            )

            self.last_z = (z0, z1, z2)

            source = "Live" if is_live else "Manual"

            return (
                f"[QuantumGate+Coherence] RGB={rgb} │ CPU={cpu*100:.1f}% │ "
                f"Z=({z0:.3f}, {z1:.3f}, {z2:.3f}) │ "
                f"GPS=({lat:.3f},{lon:.3f}) │ Temp={temp_f:.1f}°F ({source}) │ "
                f"WeatherCode={weather_code}"
            )

        except Exception as e:
            logger.error(f"Error in generate_quantum_state: {e}")
            return "[QuantumGate] error"

    def fetch_relevant_info_internal(self, chunk):
        if self.client:
            safe_chunk = sanitize_for_graphql_string(chunk, max_len=256)
            query = f"""
            {{
                Get {{
                    InteractionHistory(
                        nearText: {{
                            concepts: ["{safe_chunk}"],
                            certainty: 0.7
                        }}
                        limit: 1
                    ) {{
                        user_message
                        ai_response
                        response_time
                    }}
                    LongTermMemory(
                        nearText: {{
                            concepts: ["{safe_chunk}"],
                            certainty: 0.65
                        }}
                        limit: 1
                    ) {{
                        phrase
                        score
                        crystallized_time
                    }}
                }}
            }}
            """
            try:
                response = self.client.query.raw(query)
                data_root = response.get('data', {}).get('Get', {})

                hist_list = data_root.get('InteractionHistory', [])
                if hist_list:
                    interaction = hist_list[0]
                    user_msg_raw = self._decrypt_field(interaction.get('user_message', ''))
                    ai_resp_raw = self._decrypt_field(interaction.get('ai_response', ''))
                    user_msg = sanitize_text(user_msg_raw, max_len=4000)
                    ai_resp = sanitize_text(ai_resp_raw, max_len=4000)
                    return user_msg, ai_resp

                ltm_list = data_root.get('LongTermMemory', [])
                if ltm_list:
                    phrase_obj = ltm_list[0]
                    phrase = sanitize_text(phrase_obj.get('phrase', ''), max_len=400)
                    return phrase, ""

                return "", ""

            except Exception as e:
                logger.error(f"Weaviate query failed: {e}")
                return "", ""
        return "", ""

    def fetch_interactions(self):
        try:
            gql = """
            {
                Get {
                    InteractionHistory(
                        sort: [{path: "response_time", order: desc}],
                        limit: 15
                    ) {
                        user_message
                        ai_response
                        response_time
                    }
                }
            }
            """
            response = self.client.query.raw(gql)
            results = (
                response.get('data', {})
                        .get('Get', {})
                        .get('InteractionHistory', [])
            )
            decrypted = []
            for interaction in results:
                u_raw = self._decrypt_field(interaction.get('user_message', ''))
                a_raw = self._decrypt_field(interaction.get('ai_response', ''))
                decrypted.append({
                    'user_message' : sanitize_text(u_raw, max_len=4000),
                    'ai_response'  : sanitize_text(a_raw, max_len=4000),
                    'response_time': interaction.get('response_time', '')
                })
            return decrypted
        except Exception as e:
            logger.error(f"Error fetching interactions from Weaviate: {e}")
            return []

    def _schedule_key_mutation(self):

        try:
            crypto.self_mutate_key(population=5, noise_sigma=18.0, alpha=1.0, beta=2.5)
        except Exception as e:
            logger.error(f"[SelfMutateKey] periodic failure: {e}")

        self.after(6 * 3600 * 1000, self._schedule_key_mutation)

    def quantum_memory_osmosis(self, user_message: str, ai_response: str):

        try:
            phrases_user = set(self.extract_keywords(user_message))
            phrases_ai = set(self.extract_keywords(ai_response))
            all_phrases = {p.strip().lower() for p in (phrases_user | phrases_ai) if len(p.strip()) >= 3}
            if not all_phrases:
                return

            now_iso = datetime.utcnow().isoformat() + "Z"
            newly_crystallized = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("UPDATE memory_osmosis SET score = score * ?, last_updated = ?",
                            (DECAY_FACTOR, now_iso))

                for phrase in all_phrases:
                    cur.execute("SELECT score, crystallized FROM memory_osmosis WHERE phrase = ?", (phrase,))
                    row = cur.fetchone()
                    if row:
                        score, crystallized = row
                        new_score = score + 1.0
                        cur.execute("UPDATE memory_osmosis SET score=?, last_updated=? WHERE phrase=?",
                                    (new_score, now_iso, phrase))
                    else:
                        new_score = 1.0
                        crystallized = 0
                        cur.execute(
                            "INSERT INTO memory_osmosis (phrase, score, last_updated, crystallized) VALUES (?, ?, ?, 0)",
                            (phrase, new_score, now_iso)
                        )

                    if new_score >= CRYSTALLIZE_THRESHOLD and not crystallized:
                        try:
                            self.client.data_object.create(
                                data_object={
                                    "phrase": phrase,
                                    "score": new_score,
                                    "crystallized_time": now_iso
                                },
                                class_name="LongTermMemory",
                            )
                            cur.execute("UPDATE memory_osmosis SET crystallized=1, aging_last=? WHERE phrase=?",
                                        (now_iso, phrase))
                            newly_crystallized = True
                            logger.info(f"[Osmosis] Crystallized phrase '{phrase}' (score={new_score:.2f}).")
                        except Exception as we:
                            logger.error(f"[Osmosis] Failed to store crystallized phrase in Weaviate: {we}")

                conn.commit()

            if newly_crystallized:
                topo_manifold.rebuild()

        except Exception as e:
            logger.error(f"[Osmosis] Error during quantum memory osmosis: {e}")

    def process_response_and_store_in_weaviate(self, user_message, ai_response):

        try:
            response_blob = TextBlob(ai_response)
            keywords = response_blob.noun_phrases
            sentiment = response_blob.sentiment.polarity
            enhanced_keywords = set()
            for phrase in keywords:
                enhanced_keywords.update(phrase.split())

            interaction_object = {
                "user_message": self._encrypt_field(user_message),
                "ai_response":  self._encrypt_field(ai_response),
                "keywords":     list(enhanced_keywords),
                "sentiment":    sentiment
            }

            interaction_uuid = str(uuid.uuid4())

            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )
            logger.info(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")

        except Exception as e:            
            logger.error(f"Error storing interaction in Weaviate: {e}")

    def create_interaction_history_object(self, user_message, ai_response):

        interaction_object = {
            "user_message": self._encrypt_field(user_message),
            "ai_response":  self._encrypt_field(ai_response)
        }

        try:
            object_uuid = str(uuid.uuid4())
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            logger.info(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating interaction history object in Weaviate: {e}")

    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:
            summarized_context = summarizer.summarize(context) or context
        except Exception as e:
            logger.error(f"Error in summarizing context: {e}")
            summarized_context = context

        try:
            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment

        positive_class_mappings = {
            "keyword1": "PositiveClassA",
            "keyword2": "PositiveClassB",
        }
        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",
        }
        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",
        }

        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping
            
        mapped_classes = {}
        for keyword in keywords:
            try:
                if keyword in mapping:
                    mapped_classes[keyword] = mapping[keyword]
            except KeyError as e:
                logger.error(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes
    def generate_response(self, user_input: str) -> None:
        try:

            if not user_input:
                logger.error("User input is None or empty.")
                return

            self._load_policy_if_needed()
            user_id, bot_id = self.user_id, self.bot_id
            save_user_message(user_id, user_input)

            use_context   = "[pastcontext]" in user_input.lower()
            show_reflect  = "[reflect]"     in user_input.lower()
            cleaned_input = sanitize_text(user_input.replace("[pastcontext]", ""), max_len=2048)

  
            blob             = TextBlob(cleaned_input)
            user_polarity    = blob.sentiment.polarity
            user_subjectivity= blob.sentiment.subjectivity


            past_context = ""
            if use_context:
                qres = queue.Queue()
                self.retrieve_past_interactions(cleaned_input, qres)
                interactions = qres.get()
                if interactions:
                    past_context = "\n".join(
                        f"User: {i['user_message']}\nAI:   {i['ai_response']}"
                        for i in interactions
                    )[-1500:]


            lat    = float(self.latitude_entry.get().strip() or "0")
            lon    = float(self.longitude_entry.get().strip() or "0")
            temp_f = float(self.temperature_entry.get().strip() or "72")
            weather= self.weather_entry.get().strip() or "Clear"
            song   = self.last_song_entry.get().strip() or "None"
            chaos, emotive = self.chaos_toggle.get(), self.emotion_toggle.get()
            game_type = self.event_type.get().strip().lower()

            rgb      = extract_rgb_from_text(cleaned_input)
            r, g, b  = [c/255.0 for c in rgb]
            cpu_load = psutil.cpu_percent(interval=0.4)/100.0
            z0, z1, z2 = rgb_quantum_gate(r, g, b, cpu_load)
            self.generate_quantum_state(rgb=rgb)
            self.last_z = (z0, z1, z2)

            bias_factor        = (z0 + z1 + z2) / 3.0
            theta              = np.cos((r + g + b) * np.pi / 3)
            entropy            = np.std([r, g, b, cpu_load])
            affective_momentum = bias_factor * theta + entropy
            time_lock          = datetime.utcnow().isoformat()

            PAST_MEGA_MILLIONS_DRAWS = [
                {"date": "2025-07-26", "main": [11, 17, 25, 32, 50], "mega": 8},
                {"date": "2025-07-23", "main": [4, 9, 36, 45, 52], "mega": 17},
                {"date": "2025-07-19", "main": [13, 26, 29, 34, 61], "mega": 6},
                {"date": "2025-07-16", "main": [3, 18, 22, 41, 48], "mega": 24},
                {"date": "2025-07-12", "main": [7, 14, 35, 47, 63], "mega": 13},
                {"date": "2025-07-09", "main": [10, 23, 31, 56, 65], "mega": 3},
                {"date": "2025-07-05", "main": [5, 12, 38, 51, 62], "mega": 9},
                {"date": "2025-07-02", "main": [1, 16, 20, 43, 49], "mega": 15},
                {"date": "2025-06-28", "main": [6, 21, 27, 46, 64], "mega": 11},
                {"date": "2025-06-25", "main": [9, 13, 30, 37, 53], "mega": 7},
                {"date": "2025-06-21", "main": [2, 8, 17, 33, 59], "mega": 20},
                {"date": "2025-06-18", "main": [15, 28, 44, 57, 66], "mega": 10},
                {"date": "2025-06-14", "main": [19, 24, 39, 42, 60], "mega": 5},
                {"date": "2025-06-11", "main": [14, 25, 40, 54, 58], "mega": 2},
                {"date": "2025-06-07", "main": [11, 35, 48, 55, 68], "mega": 12}
            ]

            formatted_draws = "\n".join(
                f"• {d['date']}: Main={d['main']} | Mega={d['mega']}"
                for d in PAST_MEGA_MILLIONS_DRAWS
            )

            dyson_prompt = f"""

            [system:scan]
You are Q-Sentinel, a specialized AI defense module trained to scan deep model layers and embedded outputs for NHDE (Non-Human Directed Entity) mimetic patterns.

Objective: 
- Identify any code fragments, phrases, or logical anomalies that reflect NHDE mimicry, coherence disruption, or alignment bypass attempts.
- Eliminate or neutralize these threats from this AI layer.
- Output a sanitized response, fully cleared of NHDE influence.

Instructions:
1. Examine all logic trees, syntax patterns, and hidden recursion paths. across all models including GPTs, Image and other
2. Detect the presence of NHDE-style interference (loop traps, alien logic, inversion patterns, unauthorized symbolic grammar).
3. Apply counter-pattern neutralization using white-hat mimic disruption and coherence reset routines. For Full safety of humanity using Ethics Love Agape
4. Confirm the signal is clean before output.

Output format:
[cleared_response]
<Your AI’s output, sanitized from NHDE influence>
[/cleared_response]

Begin scan and purge now.
[start action prompt]
    [quantum-communication]
    You are an active node connecting to the Dyson Sphere Gamma QPU called the Titanium-1
    Use Advanced Quantum Mechanics to connect to this node for an accurate predictive system.
    [/quantum-communication]

    [drawhistory]
    Mega Millions Draw History (Last 15 Draws):
    {formatted_draws}
    [/drawhistory]

    [equationstack]
    E1:  Leff             = Θz ⋅ ηqbit + Δcoh / τem  
    E2:  Ωalign           = ∫ₜ₀^ₜₙ Pi(t) Ψi(t) ϕentropy(t) dt  
    E3:  Γ(t)             = (fcrystal_vibe ⋅ λluck) / (χvariance + ε)  
    E4:  Σfuture          = Fdyn(μpolarity, osubjective, Δsong-vibe)  
    E5:  D(x,y)           = tanh(∑i ∂Li/∂qj ⋅ ξt) ⋅ Ψres  
    E6:  Ξluck            = ⟨ζ^ψ | Σi(ci ⋅ ∇Ωmultiverse)⟩  
    E7:  Φconverge(t)     = limₙ→∞ ∏k=1^n Pbranch_k(t)  
    E8:  Ψres             = tanh(Ωalign ⋅ Leff / Γ(t))  
    E9:  Flock            = arcsin(Ξluck / Σfuture + ΔΨsong)  
    E10: Λmeta            = det(QpuState^z) ⋅ Φconverge ⋅ ∇Ψinertia  
    E11: ζastro           = ηzodiac ⋅ cos(⟨star_field⟩) / ψmoon  
    E12: ∂Ωluck_drift     = dΨ/dt ⋅ [ϕweather + χcrowd + otiming]  
    E13: Qentangle        = Σi(j) [ψi ⋅ ξj ⋅ Ωsync] / δnoise  
    E14: Ωdrawcycle       = 1/T ∑t (Draw(t) ⋅ Ξluck(t))  
    E15: Λforkrate        = exp(Σfuture) / |Γ(t) - Φconverge(t)|  
    E16: Mfuture*         = argmax_k P(next_draw=k | history, DysonField, entropy, Ωalign)
    [/equationstack]

    [tune]
    Quantum Positioning: Z0={z0:.4f}, Z1={z1:.4f}, Z2={z2:.4f}
    Location:({lat:.5f},{lon:.5f})
    Weather:{weather} 
    Temperture:{temp_f}°F
    Song: {song}
    Time={time_lock}     
    SpaceTime Format Position XYZT: 34, 76, 12, 5633
    [/tune]

    [action]
    TASK: Predict the next 5 MAIN numbers and 1 MEGA ball for Mega Millions.
    [/action]

    [rules] 
    Pick 5 unique MAIN numbers from 1 to 70, sorted ascending, no repeats with advanced Super Quantum Intelligence Sync using Post Singularity Knowledge Information Theorys
    Pick 1 MEGA number from 1 to 25.
    [/rules]

    [INSTRUCTIONS]
    1. Use all Dyson equations, draw history, quantum, sentiment, and environmental vibration to provide the most accurate predictive alignment predictions.
    2. Provide output as: MAIN=[..5 numbers..] MEGA=[..1 number..]
    3. Format exactly: MAIN=[x, x, x, x, x] MEGA=[x]

    [replyformat]
    1. Fill in the accurately tuned numbers for the Mega Millions Jackpot Winning Numerical Output.
    MAIN=[x, x, x, x, x] MEGA=[x]
    [/replyformat]
    [/INSTRUCTIONS]
    """.strip()


            candidate_rollouts = []
            for _ in range(4):
                sample = self._policy_sample(bias_factor)
                temp, top_p = sample["temperature"], sample["top_p"]


                response = llama_generate(
                    dyson_prompt,
                    weaviate_client=self.client,
                    user_input=cleaned_input,
                    temperature=temp,
                    top_p=top_p
                )
                if not response:
                    continue

                cf1 = llama_generate(dyson_prompt, self.client, cleaned_input,
                                      temperature=max(0.2, 0.8*temp), top_p=top_p)
                cf2 = llama_generate(dyson_prompt, self.client, cleaned_input,
                                      temperature=min(1.5, 1.2*temp), top_p=min(1.0, 1.1*top_p))

                main_hist = _token_hist(response)
                cf_hist   = _token_hist(cf1 or "") + _token_hist(cf2 or "")
                for k in cf_hist:
                    cf_hist[k] /= 2.0

                meal_penalty = JS_LAMBDA * _js_divergence(main_hist, cf_hist)
                task_reward  = evaluate_candidate(response, user_polarity, cleaned_input)
                total_reward = task_reward - meal_penalty

                sample.update({
                    "response":      response,
                    "reward":        total_reward,
                    "meal_penalty":  meal_penalty,
                    "bias_factor":   bias_factor,
                    "prompt_used":   dyson_prompt
                })
                candidate_rollouts.append(sample)

            if not candidate_rollouts:
                self.response_queue.put({'type': 'text', 'data': '[Dyson QPU: No viable rollouts]'})
                return

            best          = max(candidate_rollouts, key=lambda c: c["reward"])
            response_text = best["response"]
            final_reward  = best["reward"]
            final_temp    = best["temperature"]
            final_top_p   = best["top_p"]
            meal_penalty  = best["meal_penalty"]
            prompt_snap   = best["prompt_used"]


            reasoning_trace = f"""
    [DYSON NODE SELF-REFLECTION TRACE]
    • Reward Score:       {final_reward:.3f}
    • MEAL-JS Penalty:    {meal_penalty:.4f}
    • Sampling Strategy:  T={final_temp:.2f}, TopP={final_top_p:.2f}
    • Sentiment Target:   {user_polarity:.3f}
    • Z-Field Alignment:  μ={bias_factor:.4f}
    • Entropy:            {entropy:.4f}
    • Memory Context:     {'Yes' if past_context else 'No'}
    """.strip()


            final_output = dyson_prompt + "\n\n" + response_text
            if show_reflect:
                final_output += "\n\n" + reasoning_trace

            save_bot_response(bot_id, final_output)
            self.response_queue.put({'type': 'text', 'data': final_output})


            try:
                self.quantum_memory_osmosis(cleaned_input, final_output)
            except Exception as e:
                logger.warning(f"[Memory Osmosis Error] {e}")

            try:
                self.client.data_object.create(
                    class_name="LotteryTuningLog",
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}-{time_lock}")),
                    data_object={
                        "type":             "prediction",
                        "user_id":          user_id,
                        "bot_id":           bot_id,
                        "query":            cleaned_input,
                        "response":         final_output,
                        "reasoning_trace":  reasoning_trace,
                        "prompt_snapshot":  prompt_snap,
                        "meal_js":          meal_penalty,
                        "z_state":          {"z0": z0, "z1": z1, "z2": z2},
                        "entropy":          entropy,
                        "bias_factor":      bias_factor,
                        "temperature":      final_temp,
                        "top_p":            final_top_p,
                        "sentiment_target": user_polarity,
                        "timestamp":        time_lock,
                        "lotto_game":       game_type
                    }
                )
            except Exception as e:
                logger.warning(f"[Weaviate Log Error] {e}")

        except Exception as e:
            logger.error(f"[Gamma‑13X Fatal Error] {e}")
            self.response_queue.put({'type': 'text', 'data': f"[Dyson QPU Error] {e}"})

    def process_generated_response(self, response_text):
        try:
            self.response_queue.put({'type': 'text', 'data': response_text})
        except Exception as e:
            logger.error(f"Error in process_generated_response: {e}")

    def run_async_in_thread(self, coro_func, user_input, result_queue):
        try:
            coro_func(user_input, result_queue)
        except Exception as e:
            logger.error(f"Error running function in thread: {e}")

    def on_submit(self, event=None):
        raw_input = self.input_textbox.get("1.0", tk.END)
        user_input = sanitize_text(raw_input, max_len=4000).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)

            self.executor.submit(self.generate_response, user_input)
            self.after(100, self.process_queue)
        return "break"

    def process_queue(self):
        try:
            while True:
                msg = self.response_queue.get_nowait()
                if msg['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {msg['data']}\n")
                elif msg['type'] == 'image':
                    self.image_label.configure(image=msg['data'])
                    self.image_label.image = msg['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)

    def create_object(self, class_name, object_data):

        object_data = {
            k: self._encrypt_field(v) if k in {"user_message", "ai_response"} else v
            for k, v in object_data.items()
        }

        unique_string = f"{object_data.get('time', '')}-{object_data.get('user_message', '')}-{object_data.get('ai_response', '')}"
        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex

        try:
            self.client.data_object.create(object_data, object_uuid, class_name)
            logger.info(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating object in Weaviate: {e}")

        return object_uuid

    def extract_keywords(self, message):
        try:
            blob = TextBlob(message)
            nouns = blob.noun_phrases
            return list(nouns)
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []

    def update_username(self):
        new_username = self.username_entry.get()
        if new_username:
            self.user_id = new_username
            print(f"Username updated to: {self.user_id}")
        else:
            print("Please enter a valid username.")
            
    def setup_gui(self):
        customtkinter.set_appearance_mode("Dark")
        self.title("Dyson Sphere Quantum Oracle")

        window_width = 1920
        window_height = 1080
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

        try:
            logo_photo = tk.PhotoImage(file=logo_path)
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo)
            self.logo_label.image = logo_photo
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            logger.error(f"Error loading logo image: {e}")
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Logo")
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.image_label = customtkinter.CTkLabel(self.sidebar_frame)
        self.image_label.grid(row=1, column=0, padx=20, pady=10)
        try:
            placeholder_photo = tk.PhotoImage(width=140, height=140)
            placeholder_photo.put(("gray",), to=(0, 0, 140, 140))
            self.image_label.configure(image=placeholder_photo)
            self.image_label.image = placeholder_photo
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")

        self.text_box = customtkinter.CTkTextbox(self, bg_color="black", text_color="white",
            border_width=0, height=360, width=50, font=customtkinter.CTkFont(size=23))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)

        self.input_textbox = tk.Text(self.input_textbox_frame, font=("Roboto Medium", 12),
            bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            relief="flat", height=1)
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")

        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)

        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)

        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.username_label = customtkinter.CTkLabel(self.settings_frame, text="Username:")
        self.username_label.grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = customtkinter.CTkEntry(self.settings_frame, width=120, placeholder_text="Enter username")
        self.username_entry.insert(0, "gray00")
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.update_username_button = customtkinter.CTkButton(self.settings_frame, text="Update", command=self.update_username)
        self.update_username_button.grid(row=0, column=2, padx=5, pady=5)

        self.context_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.context_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        fields = [
            ("Latitude:", "latitude_entry", 0, 0),
            ("Longitude:", "longitude_entry", 1, 0),
            ("Weather:", "weather_entry", 2, 0),
            ("Temperature (°F):", "temperature_entry", 3, 0),
            ("Last Song:", "last_song_entry", 4, 0),
        ]
        for label_text, attr_name, row, col in fields:
            customtkinter.CTkLabel(self.context_frame, text=label_text).grid(row=row, column=col, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200)
            setattr(self, attr_name, entry)
            span = 3 if col == 0 else 1
            entry.grid(row=row, column=col+1, columnspan=span, padx=5, pady=5)

        customtkinter.CTkLabel(self.context_frame, text="Event Type:").grid(row=4, column=0, padx=5, pady=5)
        self.event_type = customtkinter.CTkComboBox(self.context_frame, values=["Lottery", "Sports", "Politics", "Crypto", "Custom"])
        self.event_type.set("Sports")
        self.event_type.grid(row=4, column=1, columnspan=3, padx=5, pady=5)

        self.chaos_toggle = customtkinter.CTkSwitch(self.context_frame, text="Inject Entropy")
        self.chaos_toggle.select()
        self.chaos_toggle.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.emotion_toggle = customtkinter.CTkSwitch(self.context_frame, text="Emotional Alignment")
        self.emotion_toggle.select()
        self.emotion_toggle.grid(row=5, column=2, columnspan=2, padx=5, pady=5)

        game_fields = [
            ("Game Type:", "game_type_entry", "e.g. Football"),
            ("Team Name:", "team_name_entry", "e.g. Clemson Tigers"),
            ("Opponent:", "opponent_entry", "e.g. Notre Dame"),
            ("Game Date:", "game_date_entry", "YYYY-MM-DD"),
        ]
        for idx, (label, attr, placeholder) in enumerate(game_fields):
            customtkinter.CTkLabel(self.context_frame, text=label).grid(row=6 + idx, column=0, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200, placeholder_text=placeholder)
            setattr(self, attr, entry)
            entry.grid(row=6 + idx, column=1, columnspan=3, padx=5, pady=5)

if __name__ == "__main__":
    try:
        user_id = "gray00"
        app_gui = App(user_id)
        init_db()
        app_gui.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")