from __future__ import annotations

import datetime as dt
import math
import tempfile
from pathlib import Path

from ...core.models import ReportStats
from ...core.rendering import BaseReportRenderer
from .models import Persona, Quote, ReportAnalysis, Topic


class ReportRenderer(BaseReportRenderer):
    def _header(self, stats: ReportStats) -> None:
        self.draw.text(
            (self.MARGIN, self.y),
            "群聊日常分析看板",
            font=self.font_title,
            fill=self.TEAL,
        )
        date_text = f"分析日期  //  {stats.target_date.isoformat()}"
        date_width = self.draw.textlength(date_text, font=self.font_small)
        self.draw.text(
            (self.WIDTH - self.MARGIN - date_width, self.y + 12),
            date_text,
            font=self.font_small,
            fill=self.TEAL,
        )
        self.y += 58
        self.draw.line(
            (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y),
            fill=self.TEAL,
            width=3,
        )
        self.y += 24

    def _metrics(self, stats: ReportStats) -> None:
        values = (
            (stats.message_count, "当日消息数", self.TEAL),
            (stats.active_member_count, "活跃成员数", self.NAVY),
            (stats.interaction_count, "互动次数", self.CORAL),
            (stats.character_count, "文本字符数", self.PINK),
        )
        gap = 14
        width = (self.WIDTH - self.MARGIN * 2 - gap * 3) // 4
        height = 110
        for index, (value, label, color) in enumerate(values):
            x = self.MARGIN + index * (width + gap)
            self._card((x, self.y, x + width, self.y + height))
            value_text = str(value)
            value_width = self.draw.textlength(value_text, font=self.font_metric)
            self.draw.text(
                (x + (width - value_width) / 2, self.y + 18),
                value_text,
                font=self.font_metric,
                fill=self.TEXT,
            )
            label_width = self.draw.textlength(label, font=self.font_small)
            self.draw.text(
                (x + (width - label_width) / 2, self.y + 67),
                label,
                font=self.font_small,
                fill=color,
            )
        self.y += height + 18

        busiest = "暂无可用时段"
        if stats.busiest_hour is not None:
            busiest = f"{stats.busiest_hour:02d}:00-{stats.busiest_hour:02d}:59"
        self.draw.rounded_rectangle(
            (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y + 88),
            radius=8,
            fill="#49c9c3",
        )
        self.draw.text(
            (self.MARGIN + 28, self.y + 14),
            "最活跃时间段",
            font=self.font_small,
            fill="#ffffff",
        )
        self.draw.text(
            (self.MARGIN + 28, self.y + 37),
            busiest,
            font=self.font_metric,
            fill="#ffffff",
        )
        self.y += 102

    def _topics(self, topics: tuple[Topic, ...]) -> None:
        self._section_title("话题总结")
        for index, topic in enumerate(topics, 1):
            content_width = self.WIDTH - self.MARGIN * 2 - 48
            summary_lines = self._wrap(
                topic.summary, self.font_body, content_width, max_lines=4
            )
            meta_parts = []
            if topic.participants:
                meta_parts.append("参与：" + "、".join(topic.participants))
            if topic.keywords:
                meta_parts.append("关键词：" + " / ".join(topic.keywords))
            meta_lines = (
                self._wrap(
                    "  |  ".join(meta_parts),
                    self.font_small,
                    content_width,
                    max_lines=2,
                )
                if meta_parts
                else []
            )
            height = 66 + len(summary_lines) * (self._line_height(self.font_body) + 4)
            if meta_lines:
                height += 12 + len(meta_lines) * (
                    self._line_height(self.font_small) + 2
                )
            self._card((self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y + height))
            self.draw.text(
                (self.MARGIN + 22, self.y + 18),
                f"#{index:02d}",
                font=self.font_h3,
                fill=self.TEAL,
            )
            self.draw.text(
                (self.MARGIN + 84, self.y + 18),
                topic.title,
                font=self.font_h3,
                fill=self.TEXT,
            )
            text_y = self._draw_lines(
                self.MARGIN + 22, self.y + 55, summary_lines, self.font_body, self.MUTED
            )
            if meta_lines:
                self._draw_lines(
                    self.MARGIN + 22,
                    text_y + 5,
                    meta_lines,
                    self.font_small,
                    self.TEAL,
                    2,
                )
            self.y += height + self.GAP

    def _personas(self, personas: tuple[Persona, ...]) -> None:
        if not personas:
            return
        self._section_title("群友画像")
        gap = 16
        width = (self.WIDTH - self.MARGIN * 2 - gap) // 2
        for row_start in range(0, len(personas), 2):
            row = personas[row_start : row_start + 2]
            prepared: list[tuple[Persona, list[str]]] = []
            heights: list[int] = []
            for persona in row:
                lines = self._wrap(
                    persona.description, self.font_small, width - 36, max_lines=5
                )
                prepared.append((persona, lines))
                heights.append(
                    72 + len(lines) * (self._line_height(self.font_small) + 2)
                )
            height = max(heights)
            for column, (persona, lines) in enumerate(prepared):
                x = self.MARGIN + column * (width + gap)
                self._card((x, self.y, x + width, self.y + height))
                self.draw.ellipse(
                    (x + 18, self.y + 17, x + 56, self.y + 55), fill=self.TEAL
                )
                initial = persona.name[:1] or "群"
                initial_width = self.draw.textlength(initial, font=self.font_h3)
                self.draw.text(
                    (x + 37 - initial_width / 2, self.y + 20),
                    initial,
                    font=self.font_h3,
                    fill="#ffffff",
                )
                self.draw.text(
                    (x + 68, self.y + 16),
                    persona.name,
                    font=self.font_h3,
                    fill=self.TEXT,
                )
                self.draw.text(
                    (x + 68, self.y + 43),
                    persona.title,
                    font=self.font_small,
                    fill=self.PINK,
                )
                self._draw_lines(
                    x + 18, self.y + 70, lines, self.font_small, self.MUTED, 2
                )
            self.y += height + self.GAP

    def _quotes(self, quotes: tuple[Quote, ...]) -> None:
        if not quotes:
            return
        self._section_title("今日金句")
        for quote in quotes:
            content_width = self.WIDTH - self.MARGIN * 2 - 44
            quote_lines = self._wrap(
                f"“{quote.text}”", self.font_body, content_width, max_lines=4
            )
            comment_lines = (
                self._wrap(quote.comment, self.font_small, content_width, max_lines=3)
                if quote.comment
                else []
            )
            height = 62 + len(quote_lines) * (self._line_height(self.font_body) + 4)
            if comment_lines:
                height += 10 + len(comment_lines) * (
                    self._line_height(self.font_small) + 2
                )
            self._card((self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y + height))
            self.draw.text(
                (self.MARGIN + 22, self.y + 16),
                quote.name,
                font=self.font_h3,
                fill=self.TEAL,
            )
            text_y = self._draw_lines(
                self.MARGIN + 22, self.y + 50, quote_lines, self.font_body, self.TEXT
            )
            if comment_lines:
                self._draw_lines(
                    self.MARGIN + 22,
                    text_y + 4,
                    comment_lines,
                    self.font_small,
                    self.PINK,
                    2,
                )
            self.y += height + self.GAP

    def _overview(self, analysis: ReportAnalysis) -> None:
        self._section_title("群聊锐评")
        title_lines = self._wrap(
            analysis.title, self.font_h3, self.WIDTH - self.MARGIN * 2 - 44, max_lines=2
        )
        category_height = math.ceil(len(analysis.categories) / 2) * 96
        comment_lines = self._wrap(
            analysis.comment,
            self.font_body,
            self.WIDTH - self.MARGIN * 2 - 76,
            max_lines=6,
        )
        height = 42 + len(title_lines) * (self._line_height(self.font_h3) + 4)
        height += 30 + category_height
        height += 28 + len(comment_lines) * (self._line_height(self.font_body) + 4) + 34
        top = self.y
        self._card((self.MARGIN, top, self.WIDTH - self.MARGIN, top + height))
        y = self._draw_lines(
            self.MARGIN + 22, top + 20, title_lines, self.font_h3, self.TEXT
        )

        bar_left, bar_right = self.MARGIN + 22, self.WIDTH - self.MARGIN - 22
        bar_top = y + 10
        cursor = bar_left
        colors = (self.CORAL, self.TEAL, self.YELLOW, self.NAVY, self.PINK)
        for index, category in enumerate(analysis.categories):
            segment = (bar_right - bar_left) * category.percent / 100
            self.draw.rectangle(
                (cursor, bar_top, cursor + segment, bar_top + 10),
                fill=colors[index % len(colors)],
            )
            cursor += segment
        y = bar_top + 26

        column_gap = 14
        column_width = (bar_right - bar_left - column_gap) // 2
        for index, category in enumerate(analysis.categories):
            column = index % 2
            row = index // 2
            x = bar_left + column * (column_width + column_gap)
            item_y = y + row * 96
            self.draw.rounded_rectangle(
                (x, item_y, x + column_width, item_y + 82), radius=6, fill="#f7fafb"
            )
            self.draw.ellipse(
                (x + 14, item_y + 17, x + 24, item_y + 27),
                fill=colors[index % len(colors)],
            )
            self.draw.text(
                (x + 32, item_y + 11),
                f"{category.name} ({category.percent}%)",
                font=self.font_small,
                fill=self.TEXT,
            )
            lines = self._wrap(
                category.description, self.font_small, column_width - 28, max_lines=2
            )
            self._draw_lines(x + 14, item_y + 38, lines, self.font_small, self.MUTED, 0)

        comment_y = y + category_height + 10
        comment_bottom = top + height - 20
        self.draw.rounded_rectangle(
            (bar_left, comment_y, bar_right, comment_bottom), radius=7, fill=self.TEAL
        )
        self._draw_lines(
            bar_left + 20, comment_y + 15, comment_lines, self.font_body, "#ffffff"
        )
        self.y = top + height

    def render(self, stats: ReportStats, analysis: ReportAnalysis) -> Path:
        self._header(stats)
        self._metrics(stats)
        self._activity(stats)
        self._topics(analysis.topics)
        self._personas(analysis.personas)
        self._quotes(analysis.quotes)
        self._overview(analysis)
        self.y += 34
        self.draw.line(
            (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y),
            fill=self.BORDER,
            width=2,
        )
        self.y += 14
        self.draw.text(
            (self.MARGIN, self.y),
            "astrbot-plugin-group-summary",
            font=self.font_small,
            fill=self.TEAL,
        )
        generated = "Generated " + dt.datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        width = self.draw.textlength(generated, font=self.font_small)
        self.draw.text(
            (self.WIDTH - self.MARGIN - width, self.y),
            generated,
            font=self.font_small,
            fill=self.MUTED,
        )
        self.y += self._line_height(self.font_small) + self.MARGIN

        cropped = self.image.crop((0, 0, self.WIDTH, self.y))
        output_dir = Path(tempfile.gettempdir()) / "astrbot_group_summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"report_{stats.target_date.isoformat()}_{dt.datetime.now().strftime('%H%M%S_%f')}.png"
        )
        cropped.save(output_path, format="PNG", compress_level=3)
        return output_path


def render_report(stats: ReportStats, analysis: ReportAnalysis) -> Path:
    return ReportRenderer().render(stats, analysis)
