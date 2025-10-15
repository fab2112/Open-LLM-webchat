# Open-LLM-webchat - Gradio interface
import settings
import gradio as gr

from gradio_modal import Modal
from utils.process_memory import clear_long_term_memory
from utils.process_knowledge import upload_file_to_base_knowledge
from utils.process_chat import get_response, stop_agent_running_stream
from utils.model_dropdown import get_owners_to_models, update_model_name_dropdown, get_full_model_name
from utils.process_session import (
    get_unique_session_id,
    clear_and_start_new_session,
    load_sessions_history,
    delete_session_from_db,
    delete_all_sessions_from_db,
    init_user_and_sessions
)

# Avatars
user_avatar = "./assets/images/user_avatar.jpg"
bot_avatar = "./assets/images/bot_avatar.gif"

# Read CSS
with open("./assets/custom_style.css", "r") as f:
    custom_css = f.read()
    

with gr.Blocks(
    title="WebChat",
    css=custom_css,
    fill_width=True,
) as demo:
    # States variables
    unique_session_id = gr.State(value=get_unique_session_id())
    username = gr.State() 
    
    # Process list of unique proprietary models
    owner_to_models = get_owners_to_models()
    owners = list(owner_to_models.keys())
    full_model_name = gr.State(value=f"{owners[0]}_{owner_to_models[owners[0]][0]}") 
  
    with gr.Row():
        
        with gr.Sidebar(width=420, open=True, position="left", elem_classes="sidebar-1"):
            
            with gr.Accordion(label="User", open=False, elem_classes="accordion-1"):
                
                gr.Image(
                    value=user_avatar,
                    elem_classes="image-1",
                    show_label=False,
                    container=False,
                )
                logged_user = gr.Markdown(
                    elem_id="markdown-2",
                )
                logout_button = gr.Button(
                    "Logout \u00A0\u00A0\u00A0➜]",
                    link="/logout",
                    elem_classes="button-8",
                )

            with gr.Accordion(label="LLM Settings", open=True, elem_classes="accordion-1"):
                
                with gr.Group(elem_classes="group-1"):
                    
                    owner_dropdown = gr.Dropdown(
                        choices=owners,
                        value=owners[0],
                        label="Select Model",
                        show_label=True,
                        interactive=True,
                        elem_classes="dropdown-1",
                    )
                    
                    model_name_dropdown = gr.Dropdown(
                        choices=owner_to_models[owners[0]],
                        value=owner_to_models[owners[0]][0],
                        show_label=False,
                        interactive=True,
                        elem_classes="dropdown-1",
                    )
                    
                    with gr.Accordion(label="Advanced", open=False, elem_classes="accordion-1"):
                        
                        model_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                            interactive=True,
                            elem_classes="slider-1"
                        )
                        
                        model_top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Top_p",
                            interactive=True,
                            elem_classes="slider-1"
                        )
                        
                        model_top_k = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.1,
                            label="Top_k",
                            interactive=True,
                            elem_classes="slider-1"
                        )
                        
                        max_tokens_number = gr.Number(
                            show_label=True,
                            elem_classes="number-1",
                            label="Max output tokens",
                            minimum=100,
                            maximum=50000,
                            step=100,
                            value=settings.MAX_TOKENS,
                        )
            
            with gr.Accordion(label="Memory", open=False, elem_classes="accordion-1"):
                
                with gr.Row(elem_classes="row-4"):
                    
                    short_memory_selector_radio = gr.Radio(
                        show_label=True,
                        value="ON",
                        choices=["ON", "OFF"],
                        label="Short-term memory",
                        elem_classes="radio-2"
                    )
                    
                    short_memory_history_runs_number = gr.Number(
                        show_label=True,
                        elem_classes="number-1",
                        label="Max history runs",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=settings.SHORT_MEM_RUNS,
                    )
                    
                with gr.Row(elem_classes="row-4"):
                    
                    long_memory_selector_radio = gr.Radio(
                        show_label=True,
                        value="ON",
                        choices=["ON", "OFF"],
                        label="Long-term memory",
                        elem_classes="radio-2"
                    )
                    
                    clear_long_memory_btn = gr.Button(
                        value="Clear",
                        elem_classes="button-7",
                    )
            
            with gr.Accordion(label="Knowledge Base", open=False, elem_classes="accordion-1"):
                
                with gr.Row(elem_classes="row-5"):
                    
                    knowledge_base_selector_radio = gr.Radio(
                        show_label=False,
                        value="OFF",
                        choices=["ON", "OFF"],
                        elem_classes="radio-2"
                    )
                
                with gr.Row(elem_classes="row-5"):
                    
                    upload_file = gr.File(
                        label="Load file",
                        elem_classes="file-1",
                        file_types=[".pdf"],
                        interactive=True,
                    )
                    
                with gr.Row():
            
                    upload_knowledge_btn = gr.Button(
                        value="Upload to base",
                        elem_classes="button-2",
                    ) 
                    
            with gr.Accordion(label="Chat Settings", open=False, elem_classes="accordion-1"):
                
                with gr.Row(elem_classes="row-5"):
                    
                    mcp_tools_radio = gr.Radio(
                        show_label=True,
                        value="ON",
                        choices=["ON", "OFF"],
                        label="MCP Tools",
                        elem_classes="radio-2"
                    )
            
                with gr.Row(elem_classes="row-5"):
                    
                    chat_event_reasoning = gr.CheckboxGroup(
                        choices=["Tool", "Agent"],  
                        value=[],                   
                        label="Reasoning",
                        elem_classes="radio-2",
                        interactive=True,
                        show_label=True
                    )
                    
                with gr.Row(elem_classes="row-5"):
                    
                    chat_event_metadata_radio = gr.Radio(
                        show_label=True,
                        value="ON",
                        choices=["ON", "OFF"],
                        label="EventTool Metadata",
                        elem_classes="radio-2"
                    )
                    
                with gr.Row(elem_classes="row-5"):
                    
                    chat_stream_radio = gr.Radio(
                        show_label=True,
                        value="ON",
                        choices=["ON", "OFF"],
                        label="Chat Stream",
                        elem_classes="radio-2"
                    )
                    
                with gr.Row(elem_classes="row-5"):
                    
                    latex_mode_radio = gr.Radio(
                        show_label=True,
                        value="OFF",
                        choices=["ON", "OFF"],
                        label="LaTeX Mode",
                        elem_classes="radio-2"
                    )
            
            with gr.Accordion(label="Manage Sessions", open=False, elem_classes="accordion-1"):
                
                with gr.Row(elem_classes="row-3"):
            
                    delete_session_btn = gr.Button(
                        value="Del session",
                        elem_classes="button-3",
                    ) 
                    
                    delete_all_sessions = gr.Button(
                        value="Del sessions",
                        elem_classes="button-4",
                    )

            with gr.Accordion(label="Session History", open=True, elem_classes="accordion-1"):
                
                session_radio = gr.Radio(
                    value=None,
                    label="Session history",
                    show_label=False,
                    interactive=True,
                    elem_classes="radio-1",
                    container=False,
                )
            
        with gr.Column(min_width=1):
            pass
        
        with gr.Column(min_width=820, elem_classes="col-1"):
                
            chatbot = gr.Chatbot(
                elem_classes="chatbot",
                type="messages",
                min_height=350,
                max_height=350,
                avatar_images=(user_avatar, bot_avatar),
                layout="bubble",
                label="CHATBOT",
                container=False,
                group_consecutive_messages=True,
                line_breaks=True
            )
            
            with gr.Group(elem_classes="group-2"):
                
                with gr.Row(elem_classes="row-1"):
                    
                    msg = gr.Textbox(
                        placeholder="Ask something...",
                        show_label=False,
                        elem_classes="textbox-1",
                        container=False,
                        scale=50,
                        max_lines=2,
                        lines=1,
                        interactive=True,
                    )
                    
                with gr.Row(elem_classes="row-2"):
                    
                    btn_new = gr.Button(
                        value="🌀",
                        elem_classes="button-5",
                    )
                    
                    btn_stop = gr.Button(
                        value="⏹️",
                        elem_classes="button-1",
                        visible=False
                    )
                    
        with gr.Column(min_width=1):
            pass
        
        with Modal(visible=False, allow_user_close=False, elem_classes="modal-1") as modal_delete_sessions:
            
            with gr.Column():
                
                with gr.Row():
            
                    gr.Markdown(
                        "## Do you want to remove all sessions?",
                        container=False,
                        elem_id="markdown-1"
                    )
                    
                with gr.Row():
                    
                    btn_cancel_modal_delete_sessions = gr.Button(
                        value="Cancel",
                        elem_classes="button-6",
                    )
                    
                    btn_confirm_modal_delete_sessions = gr.Button(
                        value="Confirm",
                        elem_classes="button-6",
                    )
                            
        with Modal(visible=False, allow_user_close=False, elem_classes="modal-1") as modal_delete_long_memory:
            
            with gr.Column():
                
                with gr.Row():
            
                    gr.Markdown(
                        "## Do you want delete all long-term memory?",
                        container=False,
                        elem_id="markdown-1"
                    )
                    
                with gr.Row():
                    
                    btn_cancel_modal_delete_long_memory = gr.Button(
                        value="Cancel",
                        elem_classes="button-6",
                    )
                    
                    btn_confirm_modal_delete_long_memory = gr.Button(
                        value="Confirm",
                        elem_classes="button-6",
                    )
    
    
    # Reload session_radio for new refresh
    demo.load(
        fn=init_user_and_sessions,
        inputs=None,
        outputs=[username, session_radio, logged_user]
    )
       
    # Handlers
    #
    # Latex radio mode
    latex_mode_radio.change(
        fn=load_sessions_history,
        inputs=[session_radio, username, chat_event_metadata_radio, latex_mode_radio],
        outputs=[chatbot, chatbot],
        show_progress="hidden"
    )
    #
    # Updates selection to ensure only 1 or none
    chat_event_reasoning.change(
        fn=lambda selected: [selected[-1]] if selected and len(selected) > 1 else selected,
        inputs=chat_event_reasoning,
        outputs=chat_event_reasoning,
        show_progress="hidden"
    )
    #
    # Chat event metadata radio button
    chat_event_metadata_radio.change(
        fn=load_sessions_history,
        inputs=[session_radio, username, chat_event_metadata_radio, latex_mode_radio],
        outputs=[chatbot, chatbot],
        show_progress="hidden"
    )
    #
    # Stop stream chat runtime button
    btn_stop.click(
        fn=stop_agent_running_stream,
    )
    #
    # Owner model dropdowns selection
    owner_dropdown.change(
        fn=lambda selected_owner: (
            update_model_name_dropdown(selected_owner),  
            get_full_model_name(selected_owner, owner_to_models[selected_owner][0])  
        ),
        inputs=owner_dropdown,
        outputs=[model_name_dropdown, full_model_name]
    )
    #
    # Name model dropdowns selection
    model_name_dropdown.change(
        fn=get_full_model_name,
        inputs=[owner_dropdown, model_name_dropdown],
        outputs=full_model_name
    )
    #
    # Modal - delete all sessions button
    delete_all_sessions.click(
        fn=lambda: Modal(visible=True), 
        inputs=None,
        outputs=modal_delete_sessions,
    )
    #
    # Modal - clear long-term memory button
    clear_long_memory_btn.click(
        fn=lambda: Modal(visible=True), 
        inputs=None,
        outputs=modal_delete_long_memory,
    )
    #
    # Chat message textbox input
    msg.submit(
        fn=get_response,
        inputs=[
            msg,
            chatbot,
            full_model_name,
            model_temperature,
            model_top_p,
            model_top_k,
            max_tokens_number,
            session_radio,
            unique_session_id,
            chat_event_metadata_radio,
            chat_stream_radio,
            upload_file,
            knowledge_base_selector_radio,
            short_memory_selector_radio,
            long_memory_selector_radio,
            short_memory_history_runs_number,
            username,
            chat_event_reasoning,
            mcp_tools_radio
        ],
        outputs=[
            msg,
            chatbot,
            session_radio,
            btn_stop,
            chatbot
        ],
        show_progress_on=[msg],
        show_progress="hidden",
    )
    #
    # New session button
    btn_new.click(
        fn=clear_and_start_new_session,
        outputs=[
            chatbot,
            msg,
            session_radio,
            unique_session_id, 
            chatbot,
            upload_file
        ],
        show_progress="hidden",
    )
    #
    # Session radio selection
    session_radio.change(
        fn=load_sessions_history,
        inputs=[session_radio, username, chat_event_metadata_radio, latex_mode_radio],
        outputs=[chatbot, chatbot],
        show_progress="hidden"
    )
    #
    # Delete selected session button
    delete_session_btn.click(
        fn=delete_session_from_db,
        inputs=[session_radio, username],
        outputs=[
            session_radio,
            chatbot,
            chatbot,
        ],
        show_progress="hidden"
    )
    #
    # Modal - confirm delete all sessions button
    btn_confirm_modal_delete_sessions.click(
        fn=delete_all_sessions_from_db,
        inputs=[username],
        outputs=[
            session_radio,
            chatbot,
            modal_delete_sessions
        ],
        show_progress="hidden"
    )
    #
    # Modal - cancel delete all sessions button
    btn_cancel_modal_delete_sessions.click(
        fn=lambda: Modal(visible=False), 
        inputs=None,
        outputs=modal_delete_sessions,
    )
    #
    # Modal - confirm delete long-term memory button
    btn_confirm_modal_delete_long_memory.click(
        fn=clear_long_term_memory,
        inputs=[username],
        outputs=[modal_delete_long_memory],
        show_progress="full"
    )
    #
    # Modal - cancel delete long-term memory button
    btn_cancel_modal_delete_long_memory.click(
        fn=lambda: Modal(visible=False), 
        inputs=None,
        outputs=modal_delete_long_memory,
    )
    #
    # Upload knowledge button
    upload_knowledge_btn.click(
        fn=upload_file_to_base_knowledge, 
        inputs=[knowledge_base_selector_radio, upload_file],
        outputs=[upload_file]
    )
    

if __name__ == "__main__":
    demo.launch(
        auth=settings.USERS,
        favicon_path=bot_avatar,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
    )