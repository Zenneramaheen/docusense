"""Generate a sample PDF (fake Acme Corp HR policy) for the DocuSense demo.

Run:  python scripts/make_sample_pdf.py
Writes: sample_docs/acme-hr-policy.pdf

The document contains concrete, unambiguous facts so demo questions have
clear answers, e.g.:
  - "How many annual leave days do employees get?"        -> 24 days
  - "How long is the probation period?"                   -> 6 months
  - "What are the core working hours?"                    -> 11am to 4pm IST
  - "How much health insurance coverage is provided?"     -> Rs. 5 lakh
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "acme-hr-policy.pdf"

# (heading, body) pairs -- one per page.
PAGES = [
    (
        "1. Probation Period",
        "All new full-time employees at Acme Corp serve a probation period of "
        "6 months from their date of joining. During probation, employment may be "
        "terminated by either party with 15 days written notice. A confirmation "
        "review is conducted by the reporting manager in the sixth month. On "
        "successful completion, the employee is confirmed in writing and becomes "
        "eligible for all regular benefits described in this policy.",
    ),
    (
        "2. Leave Policy",
        "Confirmed employees are entitled to 24 annual leave days per calendar year. "
        "Annual leave is credited quarterly at 6 days per quarter and may be carried "
        "forward up to a maximum of 10 days into the next year. In addition, every "
        "employee receives 12 sick days per year, which require a medical certificate "
        "only if the absence exceeds 3 consecutive days. New parents are entitled to "
        "26 weeks of paid parental leave, which must be availed within one year of "
        "the child's birth or adoption.",
    ),
    (
        "3. Remote Work & Working Hours",
        "Acme Corp follows a hybrid model. Employees may work remotely up to "
        "2 days per week with prior approval from their manager. Core working hours "
        "are 11am to 4pm IST, during which all employees must be available for "
        "meetings regardless of location. The standard work week is Monday to Friday. "
        "Overtime is discouraged; any overtime beyond 45 hours in a week must be "
        "pre-approved and is compensated as compensatory leave.",
    ),
    (
        "4. Benefits & Compensation",
        "Acme Corp provides group health insurance coverage of Rs. 5 lakh per employee "
        "per year, extendable to spouse and up to two children at a nominal premium. "
        "All confirmed employees are eligible for an annual performance bonus of up to "
        "15% of their fixed pay, paid with the March payroll. The company also offers "
        "a National Pension System contribution of 10% of basic pay, free annual "
        "health check-ups, and a learning budget of Rs. 25,000 per year for courses "
        "and certifications.",
    ),
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Acme Corp", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 12, "Employee HR Policy Handbook", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 10, "Effective January 2026 (sample document for demo purposes)",
             new_x="LMARGIN", new_y="NEXT", align="C")

    for heading, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, body)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
