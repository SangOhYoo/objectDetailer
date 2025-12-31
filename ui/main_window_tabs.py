# ui/main_window_tabs.py 내 AdetailerUnitWidget 클래스 수정

    def init_ui(self):
        # ... (이전 코드: Header, Prompts...) ...

        # [NEW] 4. BMAB Features (ControlNet & LoRA)
        # Source: BMAB allows attaching ControlNet to fix anatomy
        bmab_group = QGroupBox("BMAB / Anatomy Fix")
        bmab_layout = QFormLayout()

        # ControlNet Model Selector
        self.chk_controlnet = QCheckBox("Enable ControlNet (Canny/Tile)")
        self.combo_cn_model = QComboBox()
        # 실제로는 로컬 경로를 스캔해야 하지만, 예시로 고정
        self.combo_cn_model.addItems(["None", "control_v11p_sd15_canny", "control_v11f1e_sd15_tile"])
        
        # ControlNet Weight
        self.spin_cn_weight = self._create_slider_spin(0.0, 2.0, 1.0, 0.1)
        
        # LoRA Injection (Specific to this pass)
        self.combo_lora = QComboBox()
        self.combo_lora.addItems(["None", "polyhedron_skin_v1.safetensors", "hand_fixed_v2.safetensors"])
        self.spin_lora_scale = self._create_slider_spin(0.0, 1.0, 0.6, 0.05)

        bmab_layout.addRow(self.chk_controlnet)
        bmab_layout.addRow("CN Model:", self.combo_cn_model)
        bmab_layout.addRow("CN Weight:", self.spin_cn_weight)
        bmab_layout.addRow("Inject LoRA:", self.combo_lora)
        bmab_layout.addRow("LoRA Scale:", self.spin_lora_scale)
        
        bmab_group.setLayout(bmab_layout)

        # ... (이전 코드: Settings Tabs...) ...
        
        self.layout.addWidget(header_group)
        self.layout.addWidget(prompt_group)
        self.layout.addWidget(bmab_group) # Add BMAB group
        self.layout.addWidget(settings_tabs)
        self.layout.addStretch()

    def get_config(self):
        config = super().get_config() # 기존 설정 가져오기
        # BMAB 설정 추가 병합
        config.update({
            "use_controlnet": self.chk_controlnet.isChecked(),
            "cn_model": self.combo_cn_model.currentText(),
            "cn_weight": self.spin_cn_weight.value(),
            "lora_model": self.combo_lora.currentText(),
            "lora_scale": self.spin_lora_scale.value()
        })
        return config