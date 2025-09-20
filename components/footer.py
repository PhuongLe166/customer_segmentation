import streamlit as st


class Footer:
    """Reusable footer component for all pages."""

    @staticmethod
    def render(text: str | None = None) -> None:
        """Render a simple, consistent footer at the bottom of the page.

        Args:
            text: Optional custom text. If None, a sensible default is used.
        """
        default_text = (
            "Final Project - DL07_K306 - Customer Segmentation - Phạm Hồng Phát - Lê Thị Ngọc Phương"
        )
        content = text or default_text

        st.markdown(
            f"""
            <style>
            .app-footer {{
                margin-top: 18px;
                padding: 12px 8px 20px;
                border-top: 1px solid #e9eef5;
                color: #6a7a8a;
                font-size: 12px;
                text-align: center;
            }}
            .app-footer a {{
                color: #1f77b4;
                text-decoration: none;
            }}
            .app-footer a:hover {{ text-decoration: underline; }}
            </style>
            <div class="app-footer">{content}</div>
            """,
            unsafe_allow_html=True,
        )


