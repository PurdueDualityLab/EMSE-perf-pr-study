"""Draw the current study flow and its distinct analytic denominators."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def box(x, y, width, height, text, color='#edf3fa'):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0.04,rounding_size=0.08',
            facecolor=color, edgecolor='#46627d', linewidth=0.8))
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=9)

    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color='#46627d', lw=1.1))

    steps = [
        '1. GitHub mining\n1,603,213 PRs through June 1, 2026',
        '2. Published AIDev attribution rules\n78,696 agentic; 1,524,066 human-candidate; 451 unresolved',
        '3. Performance classification\n29,483 performance PRs',
        '4. Strict human-candidate filtering\n24,680 human candidates',
        '5. Quality filtering and deterministic weekly sampling\n2,260 PRs: 1,130 per arm across 65 ISO weeks',
    ]
    for i, text in enumerate(steps):
        y = 10.7 - i*1.15
        box(.3, y, 9.4, .85, text)
        if i:
            arrow((5, y+1.15), (5, y+.85))
    box(.2, 3.55, 3.0, 1.7, 'RQ1: pattern consensus\n2,260 classified\n2,083 resolved\n177 unresolved')
    box(3.5, 3.55, 3.0, 1.7, 'RQ2: validation consensus\n2,258 classified\n1,839 positive\n1,707 resolved types')
    box(6.8, 3.55, 3.0, 1.7, 'Structural comparison\nStored base/head code\n2,029 complete cases\nLizard metrics')
    for x in (1.7, 5, 8.3):
        arrow((x, 6.1), (x, 5.25))
    box(.6, .65, 5.5, 1.65,
        'RQ3: joint pattern / validation / metric analysis\n2,081 analytic PRs\n1,699 positive validation consensuses\n1,581 resolved types', '#e5f1e8')
    arrow((1.7, 3.55), (2.2, 2.3))
    arrow((5, 3.55), (4.5, 2.3))
    ax.text(7.9, 1.5, 'Human-candidate denotes\nabsence of selected\nobservable agentic signals,\nnot confirmed authorship.',
        ha='center', va='center', fontsize=8, color='#454545')
    fig.tight_layout(pad=.3)
    target = Path(__file__).resolve().parents[1] / 'report/figures/methodology_current.pdf'
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
