from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    title: str
    summary: str
    participants: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Persona:
    name: str
    title: str
    description: str


@dataclass(frozen=True)
class Quote:
    name: str
    text: str
    comment: str


@dataclass(frozen=True)
class Category:
    name: str
    percent: int
    description: str


@dataclass(frozen=True)
class ReportAnalysis:
    title: str
    topics: tuple[Topic, ...] = ()
    personas: tuple[Persona, ...] = ()
    quotes: tuple[Quote, ...] = ()
    categories: tuple[Category, ...] = ()
    comment: str = ""
    source: str = "model"
