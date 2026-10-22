import streamlit as st


def render_chat_box():
    history = st.session_state.get("chat_history", [])

    for turn in history:
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

            # Render Source Citations
            sources = turn.get("sources", [])
            if sources:
                st.markdown("**📌 Sources:**")
                badges = ""
                for src in sources:
                    badges += (
                        f'<span style="background-color: #1e3a5f; color: #7ec8e3; padding: 3px 8px; '
                        f'border-radius: 4px; font-size: 12px; margin-right: 5px;">'
                        f'📄 {src.get("source")} · p.{src.get("page")} · {src.get("section")}'
                        f'</span>'
                    )
                st.markdown(badges, unsafe_allow_html=True)

            # Render Faithfulness Score Box
            faith_score = turn.get("faithfulness_score")
            if faith_score is not None:
                if faith_score >= 0.75:
                    emoji, label, color = "🟢", "High — well grounded", "#0f1f0f"
                elif faith_score >= 0.45:
                    emoji, label, color = "🟡", "Medium — mostly grounded", "#1f1800"
                else:
                    emoji, label, color = "🔴", "Low — verify answer manually", "#1f0f0f"

                st.markdown(
                    f'<div style="background-color: {color}; border-left: 3px solid #2ecc71; '
                    f'padding: 8px 12px; border-radius: 4px; font-size: 13px; margin-top: 8px;">'
                    f'{emoji} <strong>Faithfulness: {faith_score:.0%}</strong> — {label}'
                    f'</div>',
                    unsafe_allow_html=True
                )