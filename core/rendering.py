from __future__ import annotations

from pathlib import Path

from .config import FONT_DIR
from .models import ReportStats

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


class BaseReportRenderer:
    WIDTH = 1200

    MARGIN = 56

    GAP = 18

    BACKGROUND = "#f2fbfb"

    SURFACE = "#ffffff"

    BORDER = "#d8eeee"

    TEXT = "#263747"

    MUTED = "#687985"

    TEAL = "#32c4bd"

    CORAL = "#ff7378"

    YELLOW = "#f5cd55"

    PINK = "#ef79ad"

    NAVY = "#286273"

    def __init__(self) -> None:
        if Image is None or ImageDraw is None or ImageFont is None:
            raise RuntimeError("Pillow 未安装，无法生成日报图片")
        self.image = Image.new("RGB", (self.WIDTH, 6400), self.BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.font_title = self._font(36, bold=True)
        self.font_h2 = self._font(25, bold=True)
        self.font_h3 = self._font(21, bold=True)
        self.font_body = self._font(18)
        self.font_small = self._font(15)
        self.font_metric = self._font(31, bold=True)
        self.y = self.MARGIN

    def _font_candidates(self, bold: bool) -> list[Path]:
        local = FONT_DIR
        bundled = local / "NotoSansSC-Variable.ttf"
        if bold:
            return [
                bundled,
                local / "NotoSansCJKsc-Bold.otf",
                Path("C:/Windows/Fonts/msyhbd.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
            ]
        return [
            bundled,
            local / "NotoSansCJKsc-Regular.otf",
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
        ]

    @staticmethod
    def _font_supports_chinese(font) -> bool:
        try:
            chinese = font.getmask("群")
            missing = font.getmask("\u0378")
            return chinese.size != missing.size or bytes(chinese) != bytes(missing)
        except Exception:
            return False

    def _font(self, size: int, bold: bool = False):
        for candidate in self._font_candidates(bold):
            if not candidate.is_file():
                continue
            try:
                font = ImageFont.truetype(str(candidate), size=size)
                if candidate.name == "NotoSansSC-Variable.ttf":
                    try:
                        font.set_variation_by_name("Bold" if bold else "Regular")
                    except (AttributeError, OSError):
                        pass
                if self._font_supports_chinese(font):
                    return font
            except Exception:
                continue
        raise RuntimeError("未找到可用的中文字体，无法生成日报图片")

    def _line_height(self, font) -> int:
        box = self.draw.textbbox((0, 0), "Ag群", font=font)
        return box[3] - box[1] + 8

    def _wrap(
        self, text: str, font, width: int, max_lines: int | None = None
    ) -> list[str]:
        paragraphs = str(text or "").replace("\r", "").split("\n")
        lines: list[str] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not paragraph:
                lines.append("")
                if max_lines and len(lines) >= max_lines:
                    return lines
            offset = 0
            while offset < len(paragraph):
                # 用指数扩展和二分查找测量整段文字，保留字体字距并减少测量次数。
                lower, upper = 0, min(64, len(paragraph) - offset)
                while (
                    self.draw.textlength(paragraph[offset : offset + upper], font=font)
                    <= width
                ):
                    lower = upper
                    if offset + upper == len(paragraph):
                        break
                    upper = min(upper * 2, len(paragraph) - offset)
                while lower < upper:
                    middle = (lower + upper + 1) // 2
                    if (
                        self.draw.textlength(
                            paragraph[offset : offset + middle], font=font
                        )
                        <= width
                    ):
                        lower = middle
                    else:
                        upper = middle - 1
                length = max(1, lower)
                lines.append(paragraph[offset : offset + length])
                offset += length
                if max_lines and len(lines) >= max_lines:
                    if offset < len(paragraph) or paragraph_index < len(paragraphs) - 1:
                        tail = lines[-1].rstrip("。；，,. ")
                        while (
                            tail
                            and self.draw.textlength(tail + "...", font=font) > width
                        ):
                            tail = tail[:-1]
                        lines[-1] = tail + "..."
                    return lines
        return lines or [""]

    def _draw_lines(
        self, x: int, y: int, lines: list[str], font, fill: str, line_gap: int = 4
    ) -> int:
        line_height = self._line_height(font)
        for line in lines:
            self.draw.text((x, y), line, font=font, fill=fill)
            y += line_height + line_gap
        return y

    def _section_title(self, title: str) -> None:
        self.y += 24
        self.draw.rounded_rectangle(
            (self.MARGIN, self.y + 2, self.MARGIN + 6, self.y + 30),
            radius=3,
            fill=self.TEAL,
        )
        self.draw.text(
            (self.MARGIN + 18, self.y), title, font=self.font_h2, fill=self.TEAL
        )
        self.y += 44
        self.draw.line(
            (self.MARGIN, self.y, self.WIDTH - self.MARGIN, self.y),
            fill=self.BORDER,
            width=2,
        )
        self.y += 16

    def _card(self, box: tuple[int, int, int, int], fill: str | None = None) -> None:
        self.draw.rounded_rectangle(
            box, radius=8, fill=fill or self.SURFACE, outline=self.BORDER, width=2
        )

    def _activity(self, stats: ReportStats) -> None:
        self._section_title("24 小时活动")
        x1, x2 = self.MARGIN, self.WIDTH - self.MARGIN
        height = 230
        self._card((x1, self.y, x2, self.y + height))
        chart_left, chart_right = x1 + 35, x2 - 25
        baseline = self.y + 178
        chart_height = 125
        max_count = max(stats.hourly_counts) if stats.hourly_counts else 0
        bar_gap = 7
        bar_width = (chart_right - chart_left - bar_gap * 23) / 24
        palette = (self.TEAL, "#83ddd7", self.YELLOW, self.PINK, self.CORAL)
        for hour, count in enumerate(stats.hourly_counts):
            bar_height = (
                2 if max_count == 0 else max(2, round(count / max_count * chart_height))
            )
            x = chart_left + hour * (bar_width + bar_gap)
            color = (
                palette[min(4, round((count / max_count) * 4))]
                if max_count
                else self.BORDER
            )
            self.draw.rounded_rectangle(
                (x, baseline - bar_height, x + bar_width, baseline),
                radius=3,
                fill=color,
            )
            if count:
                label = str(count)
                label_width = self.draw.textlength(label, font=self.font_small)
                self.draw.text(
                    (x + (bar_width - label_width) / 2, baseline - bar_height - 22),
                    label,
                    font=self.font_small,
                    fill=self.MUTED,
                )
            if hour % 2 == 0:
                label = f"{hour:02d}时"
                label_width = self.draw.textlength(label, font=self.font_small)
                self.draw.text(
                    (x + (bar_width - label_width) / 2, baseline + 11),
                    label,
                    font=self.font_small,
                    fill=self.MUTED,
                )
        self.y += height
