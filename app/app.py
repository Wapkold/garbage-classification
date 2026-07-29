"""Gradio-демо: загрузка изображения -> предсказанный класс и вероятности.

Ключевое требование к этому файлу — предобработка обязана совпадать с той,
что была на валидации. Поэтому трансформы не пишутся заново, а берутся из
src.dataset.get_eval_transforms(): любое расхождение (другой ресайз, забытая
нормализация) даёт молчаливую деградацию — модель работает, но хуже, и понять
это по интерфейсу невозможно.

Запуск:
    python app/app.py
    python app/app.py --checkpoint checkpoints/resnet18_finetune/best.pt
    python app/app.py --share          # временная публичная ссылка
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DATA_ROOT, DEFAULT_VARIANT, IMG_SIZE, get_eval_transforms
from src.model import load_checkpoint

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# На Hugging Face Space обучения нет: рядом с app.py лежат один model.pt и
# папка examples/. Локально их нет, и оба ресурса берутся из проекта. Один и
# тот же файл работает в обоих случаях — копия app.py для деплоя неизбежно
# разъехалась бы с оригиналом.
APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL = APP_DIR / "model.pt"
LOCAL_EXAMPLES = APP_DIR / "examples"

# класс -> (эмодзи, название, категория сбора, что делать)
# Правила раздельного сбора отличаются от города к городу, поэтому формулировки
# намеренно общие — это подсказка, а не регламент.
CLASS_INFO: dict[str, tuple[str, str, str, str]] = {
    "battery": ("🔋", "Батарейки", "Опасные отходы",
                "Не в общий бак. Пункт приёма — часто в супермаркетах и магазинах электроники."),
    "biological": ("🍂", "Органика", "Пищевые отходы",
                   "Компост или отдельный бак для органики."),
    "cardboard": ("📦", "Картон", "Вторсырьё",
                  "Расправить и сложить, снять скотч. Мокрый и жирный — в общий бак."),
    "clothes": ("👕", "Одежда", "Текстиль",
                "Пункт приёма вещей или контейнер для одежды. Пригодное — на благотворительность."),
    "glass": ("🍷", "Стекло", "Вторсырьё",
              "Бак для стекла, без крышек. Керамика, зеркала и лампы туда не идут."),
    "metal": ("🥫", "Металл", "Вторсырьё",
              "Бак для металла. Банки лучше сполоснуть и смять."),
    "paper": ("📄", "Бумага", "Вторсырьё",
              "Сухая и чистая — в бумагу. Чеки, салфетки и жирная бумага — в общий бак."),
    "plastic": ("🧴", "Пластик", "Вторсырьё",
                "Смотрите маркировку 1–7: принимают не все. Сполоснуть, крышку сдать отдельно."),
    "shoes": ("👟", "Обувь", "Текстиль",
              "Пункт приёма или спецконтейнер. В бумагу и пластик нельзя."),
    "trash": ("🗑️", "Прочее", "Не перерабатывается",
              "Общий бак для смешанных отходов."),
}

CSS = """
/* Цвета берём из токенов Gradio там, где можно, — тогда тёмная тема
   работает сама. Свои переменные заводим только под акцент. */
:root {
  --acc: #2a78d6;
  --acc-2: #1baf7a;
  --acc-soft: rgba(42, 120, 214, 0.10);
}
.dark {
  --acc: #3987e5;
  --acc-2: #199e70;
  --acc-soft: rgba(57, 135, 229, 0.16);
}

.gradio-container {
  max-width: 1120px !important;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.gradio-container .prose :is(h1, h2, h3) { margin: 0; }

/* ---------- шапка ---------- */
#hero { padding: 1.75rem 0 1.5rem; text-align: center; }
#hero .eyebrow {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--acc); background: var(--acc-soft);
  padding: 0.32rem 0.7rem; border-radius: 100px; margin-bottom: 0.85rem;
}
#hero h1 {
  font-size: clamp(1.85rem, 4vw, 2.6rem); font-weight: 700;
  letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 0.6rem;
  color: var(--body-text-color);
}
#hero p {
  margin: 0 auto; max-width: 34rem; font-size: 1rem; line-height: 1.55;
  color: var(--body-text-color-subdued);
}

/* ---------- карточка результата ---------- */
#result .card {
  border: 1px solid var(--border-color-primary);
  border-radius: 18px;
  background: var(--background-fill-primary);
  overflow: hidden;
}

#result .verdict {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.35rem 1.5rem;
  border-bottom: 1px solid var(--border-color-primary);
  background: linear-gradient(180deg, var(--acc-soft), transparent);
}
#result .verdict .emoji {
  font-size: 2.4rem; line-height: 1; flex: none;
  filter: saturate(1.05);
}
#result .verdict .who { flex: 1 1 auto; min-width: 0; }
#result .verdict .cat {
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--acc); margin-bottom: 0.2rem;
}
#result .verdict .name {
  font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--body-text-color); line-height: 1.15;
}
#result .verdict .score { flex: none; text-align: right; }
#result .verdict .score .num {
  font-size: 2rem; font-weight: 700; letter-spacing: -0.03em;
  color: var(--body-text-color); line-height: 1;
}
#result .verdict .score .num span { font-size: 1.1rem; font-weight: 600; opacity: 0.55; }
#result .verdict .score .cap {
  font-size: 0.72rem; color: var(--body-text-color-subdued); margin-top: 0.2rem;
}

/* Полосы вероятностей. Один цвет на все — это номинальные категории,
   а не шкала величины; выделен только верхний вариант, остальные приглушены. */
#result .bars { padding: 1.1rem 1.5rem 1.25rem; display: grid; gap: 0.6rem; }
#result .row {
  display: grid; grid-template-columns: 8.5rem 1fr 3.2rem;
  align-items: center; gap: 0.75rem;
}
#result .row .nm {
  font-size: 0.88rem; color: var(--body-text-color);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#result .row .nm .en {
  display: block; font-size: 0.7rem; color: var(--body-text-color-subdued);
  letter-spacing: 0.02em;
}
#result .row .track {
  height: 8px; border-radius: 100px;
  background: var(--background-fill-secondary); overflow: hidden;
}
#result .row .fill {
  height: 100%; border-radius: 100px; background: var(--acc); opacity: 0.32;
  transition: width 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
#result .row.top .fill { opacity: 1; }
#result .row.top .nm { font-weight: 600; }
#result .row .pct {
  font-size: 0.82rem; text-align: right; font-variant-numeric: tabular-nums;
  color: var(--body-text-color-subdued);
}
#result .row.top .pct { color: var(--body-text-color); font-weight: 600; }

/* ---------- подсказка по утилизации ---------- */
#result .hint {
  display: flex; gap: 0.75rem; align-items: flex-start;
  padding: 1rem 1.5rem 1.25rem;
  border-top: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
}
#result .hint .mark {
  flex: none; width: 22px; height: 22px; border-radius: 100px;
  background: var(--acc-2); color: #fff; font-size: 0.8rem; font-weight: 700;
  display: grid; place-items: center; margin-top: 0.1rem;
}
#result .hint .txt {
  font-size: 0.9rem; line-height: 1.5; color: var(--body-text-color-subdued);
}

/* ---------- предупреждение о низкой уверенности ---------- */
#result .warn {
  margin: 0 1.5rem 1.25rem; padding: 0.7rem 0.9rem;
  border: 1px dashed var(--border-color-primary); border-radius: 12px;
  font-size: 0.84rem; line-height: 1.45; color: var(--body-text-color-subdued);
}

/* ---------- пустое состояние ---------- */
#result .empty {
  border: 1px dashed var(--border-color-primary); border-radius: 18px;
  padding: 3.25rem 1.5rem; text-align: center;
  color: var(--body-text-color-subdued);
}
#result .empty .ico { font-size: 2rem; opacity: 0.5; margin-bottom: 0.6rem; }
#result .empty .t { font-size: 0.95rem; }

/* ---------- подвал ---------- */
#meta {
  text-align: center; padding: 1.5rem 0 0.5rem;
  font-size: 0.78rem; color: var(--body-text-color-subdued);
}
#meta code {
  background: var(--background-fill-secondary); padding: 0.1rem 0.35rem;
  border-radius: 5px; font-size: 0.94em;
}
#meta .dot { opacity: 0.4; margin: 0 0.45rem; }

@media (max-width: 640px) {
  #result .row { grid-template-columns: 6.5rem 1fr 2.8rem; }
  #result .verdict { padding: 1.1rem; }
  #result .bars, #result .hint { padding-left: 1.1rem; padding-right: 1.1rem; }
}
"""

EMPTY_STATE = (
    '<div class="empty">'
    '<div class="ico">♻️</div>'
    '<div class="t">Загрузите фотографию или выберите пример ниже</div>'
    "</div>"
)


# --- Выбор чекпоинта -------------------------------------------------------


def find_best_checkpoint() -> Path:
    """Лучший чекпоинт среди всех прогонов — по val accuracy из самого файла.

    Прогонов в checkpoints/ обычно несколько (разные архитектуры, разное число
    эпох), и помнить, какой из них удачнее, человеку не обязательно.
    """
    if LOCAL_MODEL.exists():
        return LOCAL_MODEL  # развёрнутый Space: выбирать не из чего

    candidates = sorted(CHECKPOINT_DIR.glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"Не найдено ни одного чекпоинта в {CHECKPOINT_DIR}/*/best.pt\n"
            "Сначала обучите модель: python -m src.train"
        )

    def val_acc(path: Path) -> float:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return float(payload.get("metrics", {}).get("val_acc", 0.0))

    return max(candidates, key=val_acc)


def collect_examples(limit_per_class: int = 1) -> list[str]:
    """По одному примеру на класс — чтобы демо можно было потыкать без своих файлов."""
    if LOCAL_EXAMPLES.is_dir():
        return [str(p) for p in sorted(LOCAL_EXAMPLES.glob("*.jpg"))]

    data_dir = DATA_ROOT / DEFAULT_VARIANT
    if not data_dir.is_dir():
        return []

    examples = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        examples.extend(str(f) for f in sorted(class_dir.glob("*.jpg"))[:limit_per_class])
    return examples


# --- Отрисовка результата --------------------------------------------------


def fmt_pct(p: float) -> str:
    """Проценты без мусорных знаков: «100%», «87%», «0.4%», «<0.1%».

    Форматирование одним шаблоном давало бы «100.0%» рядом с «0.0%» —
    первое выглядит неряшливо, второе прямо неверно (вероятность не ноль).
    """
    if p >= 0.995:
        return "100%"
    if p >= 0.095:
        return f"{p:.0%}"
    if p >= 0.001:
        return f"{p * 100:.1f}%"
    return "<0.1%"


def render_result(scores: dict[str, float], top_n: int = 4) -> str:
    """Карточка результата целиком: вердикт, полосы вероятностей, подсказка.

    Собственный HTML вместо gr.Label — ради вердикта крупным планом и
    подсказки по утилизации рядом с ним; штатный компонент показывает
    только голый список вероятностей.
    """
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    top_class, top_p = ordered[0]
    emoji, name, category, hint = CLASS_INFO.get(top_class, ("❓", top_class, "", ""))

    rows = []
    for i, (cls, p) in enumerate(ordered):
        _, ru_name, _, _ = CLASS_INFO.get(cls, ("", cls, "", ""))
        rows.append(
            f'<div class="row{" top" if i == 0 else ""}">'
            f'<div class="nm">{html.escape(ru_name)}<span class="en">{html.escape(cls)}</span></div>'
            # Минимум 1.5% ширины, иначе почти нулевая вероятность рисуется
            # невидимой полоской и строка выглядит как сбой отрисовки.
            f'<div class="track"><div class="fill" style="width:{max(p * 100, 1.5):.1f}%"></div></div>'
            f'<div class="pct">{fmt_pct(p)}</div>'
            f"</div>"
        )

    # Порог 0.55 — ниже него у модели обычно два конкурирующих кандидата
    # (типичная пара: стекло и пластик), и выдавать один ответ как факт нечестно.
    warning = ""
    if top_p < 0.55:
        runner = CLASS_INFO.get(ordered[1][0], ("", ordered[1][0], "", ""))[1]
        warning = (
            f'<div class="warn">Модель не уверена: близкий вариант — '
            f"<b>{html.escape(runner)}</b> ({fmt_pct(ordered[1][1])}). "
            f"Стоит сфотографировать предмет крупнее или при другом освещении.</div>"
        )

    return (
        '<div class="card">'
        '<div class="verdict">'
        f'<div class="emoji">{emoji}</div>'
        f'<div class="who"><div class="cat">{html.escape(category)}</div>'
        f'<div class="name">{html.escape(name)}</div></div>'
        f'<div class="score"><div class="num">{top_p * 100:.0f}<span>%</span></div>'
        f'<div class="cap">уверенность</div></div>'
        "</div>"
        f'<div class="bars">{"".join(rows)}</div>'
        f"{warning}"
        f'<div class="hint"><div class="mark">i</div>'
        f'<div class="txt">{html.escape(hint)}</div></div>'
        "</div>"
    )


# --- Сборка интерфейса -----------------------------------------------------


def make_theme() -> gr.themes.Base:
    """Тема и CSS передаются в launch(), а не в Blocks.

    В Gradio 6 эти параметры переехали из конструктора Blocks в launch(), и при
    передаче в Blocks просто игнорируются — с одним лишь UserWarning в логе.
    Оформление при этом молча остаётся дефолтным.
    """
    # font= намеренно не задаём: список строк ломает сравнение тем внутри
    # Gradio 6 (fonts.py __eq__ ожидает объект Font, получает str) и приложение
    # падает при старте. Шрифт задан в CSS — там же, где остальная типографика.
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
    )


def build_demo(checkpoint: Path, device: torch.device) -> gr.Blocks:
    model, classes, payload = load_checkpoint(checkpoint, device=device)
    transform = get_eval_transforms(IMG_SIZE)

    arch = payload.get("arch", "?")
    val_acc = payload.get("metrics", {}).get("val_acc")
    # Имя прогона, а не архитектуры: у замороженного backbone и полного
    # fine-tuning класс один и тот же (resnet18_transfer), и по нему не понять,
    # какая из двух моделей загружена. Имя прогона это различает.
    run_name = payload.get("run_name") or checkpoint.parent.name

    @torch.no_grad()
    def predict(image) -> str:
        if image is None:
            return EMPTY_STATE
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
        return render_result({name: float(p) for name, p in zip(classes, probs)})

    quality = f"{val_acc:.1%}" if val_acc is not None else "—"

    with gr.Blocks(title="Куда выбросить? · классификация отходов") as demo:
        gr.HTML(
            "<div id='hero'>"
            "<div class='eyebrow'>♻️ Классификация отходов</div>"
            "<h1>Куда это выбросить?</h1>"
            f"<p>Сфотографируйте предмет — нейросеть определит тип отхода из "
            f"{len(classes)} категорий и подскажет, куда его сдать.</p>"
            "</div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image_input = gr.Image(
                    type="pil",
                    label="Фотография",
                    height=360,
                    sources=["upload", "clipboard"],
                )
                with gr.Row():
                    clear_btn = gr.ClearButton(value="Очистить")
                    submit_btn = gr.Button("Определить", variant="primary")

            with gr.Column(scale=5):
                result = gr.HTML(EMPTY_STATE, elem_id="result")

        examples = collect_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
                outputs=result,
                fn=predict,
                label="Попробуйте на примерах",
                examples_per_page=10,
            )

        gr.HTML(
            f"<div id='meta'>Модель <code>{arch}</code>"
            f"<span class='dot'>·</span>прогон <code>{run_name}</code>"
            f"<span class='dot'>·</span>точность {quality} на валидации"
            f"<span class='dot'>·</span>вычисления на <code>{device}</code></div>"
        )

        # Считаем и при загрузке файла, и по кнопке — оба пути к одной функции.
        image_input.change(predict, inputs=image_input, outputs=result)
        submit_btn.click(predict, inputs=image_input, outputs=result)
        clear_btn.add([image_input])
        clear_btn.click(lambda: EMPTY_STATE, outputs=result)

    return demo


# --- Точка входа -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio-демо классификации мусора")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="путь к чекпоинту; по умолчанию — лучший по val accuracy среди checkpoints/*/best.pt",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="публичная ссылка через туннель gradio")
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер при старте")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
    else:
        checkpoint = find_best_checkpoint()

    print(f"чекпоинт:   {checkpoint}")
    print(f"устройство: {device}")

    demo = build_demo(checkpoint, device)

    # На Hugging Face Space (переменная SPACE_ID) нужно слушать все интерфейсы,
    # а браузер открывать некому — там stdout не терминал.
    on_space = bool(os.environ.get("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0" if on_space else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser and sys.stdout.isatty(),
        theme=make_theme(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
