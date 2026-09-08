from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

from ...core.rendering import BaseReportRenderer
from ...core.statistics import compute_stats
from ...core.text import clean_text
from .domain import time_range
from .models import PersonalAnalysis, PersonalReport


class PersonalReportRenderer(BaseReportRenderer):
    def _ensure_space(self, height: int) -> None:
        if self.y + height <= self.image.height:
            return
        from PIL import Image, ImageDraw

        expanded = Image.new(
            "RGB",
            (self.WIDTH, max(self.image.height * 2, self.y + height)),
            self.BACKGROUND,
        )
        expanded.paste(self.image, (0, 0))
        self.image = expanded
        self.draw = ImageDraw.Draw(self.image)

    def _paragraph(self, text: str, font=None, color: str | None = None) -> None:
        font = font or self.font_body
        lines = self._wrap(text, font, self.WIDTH - self.MARGIN * 2)
        self._ensure_space(len(lines) * (self._line_height(font) + 4) + 20)
        self.y = (
            self._draw_lines(self.MARGIN, self.y, lines, font, color or self.TEXT) + 12
        )

    def render_personal(
        self, report: PersonalReport, analysis: PersonalAnalysis
    ) -> Path:
        self._paragraph(
            "个人聊天详情" if report.detailed else "个人聊天记录",
            self.font_title,
            self.TEAL,
        )
        self._paragraph(
            f"{clean_text(report.target_name, 80)}（{clean_text(report.target_id, 80)}）",
            self.font_h2,
        )
        self._paragraph(
            f"{report.mode} | {time_range(report.messages[0].created_at, report.messages[-1].created_at)}",
            self.font_small,
            self.MUTED,
        )
        self.draw.line(
            (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y),
            fill=self.TEAL,
            width=3,
        )
        self.y += 20
        stats = compute_stats(
            report.messages,
            dt.datetime.fromtimestamp(report.messages[0].created_at).date(),
        )
        values = (
            (len(report.messages), "个人消息", self.TEAL),
            (
                sum(period.session_count for period in report.periods),
                "活跃时段",
                self.CORAL,
            ),
            (stats.interaction_count, "提及与回复", self.PINK),
            (stats.character_count, "文本字符", self.NAVY),
        )
        width = (self.WIDTH - self.MARGIN * 2) // 4
        for index, (value, label, color) in enumerate(values):
            x = self.MARGIN + index * width
            self.draw.text((x, self.y), str(value), font=self.font_metric, fill=color)
            self.draw.text(
                (x, self.y + 46), label, font=self.font_small, fill=self.MUTED
            )
        self.y += 80
        self._paragraph(analysis.overview)
        self._activity(stats)
        self._ensure_space(110)
        self._section_title("分时段发言与上下文")
        for index, (period, detail) in enumerate(
            zip(report.periods, analysis.periods), 1
        ):
            number = report.period_number if report.detailed else index
            self._paragraph(
                f"{number:02d}  {time_range(period.start, period.end)}",
                self.font_h3,
                self.TEAL,
            )
            self._paragraph(
                f"个人 {len(period.messages)} 条 / 邻近 {len(period.context)} 条 / 活跃时段 {period.session_count} 个",
                self.font_small,
                self.MUTED,
            )
            self._paragraph(detail.title, self.font_h3)
            self._paragraph("个人发言：" + detail.summary)
            self._paragraph("讨论上下文：" + detail.context, color=self.MUTED)
            if detail.quote:
                self._paragraph("本人原话：" + detail.quote, color=self.NAVY)
            self._ensure_space(40)
            self.draw.line(
                (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y),
                fill=self.BORDER,
                width=2,
            )
            self.y += 24
        if report.detailed:
            self._ensure_space(110)
            self._section_title(f"原始记录 | 第 {report.page}/{report.page_count} 页")
            for message in report.transcript:
                label = "本人" if message.sender_id == report.target_id else "群友"
                color = (
                    self.TEAL if message.sender_id == report.target_id else self.MUTED
                )
                self._paragraph(
                    f"{dt.datetime.fromtimestamp(message.created_at):%m-%d %H:%M:%S}  "
                    f"{clean_text(message.sender_name, 80)}（{label}）",
                    self.font_small,
                    color,
                )
                self._paragraph(message.text)
        self._paragraph(
            "时间采用服务器本地时区；仅涵盖插件保留记录。邻近消息不一定构成直接回复。",
            self.font_small,
            self.MUTED,
        )
        if any(period.session_count > 1 for period in report.periods):
            self._paragraph(
                "活跃时段较多，已按时间顺序合并展示；统计覆盖本次全部个人记录。",
                self.font_small,
                self.MUTED,
            )
        source = {
            "statistics": "统计与摘录",
            "mixed": "模型分析与统计补充",
            "model": "模型分析",
        }[analysis.source]
        self._paragraph(
            f"{source} | astrbot-plugin-group-summary", self.font_small, self.TEAL
        )
        self._ensure_space(self.MARGIN)
        output_dir = Path(tempfile.gettempdir()) / "astrbot_group_summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"personal_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        )
        self.image.crop((0, 0, self.WIDTH, self.y + self.MARGIN)).save(
            output_path, format="PNG", compress_level=3
        )
        return output_path


def render_personal_report(report: PersonalReport, analysis: PersonalAnalysis) -> Path:
    return PersonalReportRenderer().render_personal(report, analysis)
