package tables

import (
	"encoding/json"
	"time"

	"gorm.io/gorm"
)

// TableClientConfig represents global client configuration in the database
type TableClientConfig struct {
	ID                      uint   `gorm:"primaryKey;autoIncrement" json:"id"`
	DropExcessRequests      bool   `gorm:"default:false" json:"drop_excess_requests"`
	PrometheusLabelsJSON    string `gorm:"type:text" json:"-"` // JSON serialized []string
	AllowedOriginsJSON      string `gorm:"type:text" json:"-"` // JSON serialized []string
	InitialPoolSize         int    `gorm:"default:300" json:"initial_pool_size"`
	EnableLogging           bool   `gorm:"" json:"enable_logging"`
	EnableGovernance        bool   `gorm:"" json:"enable_governance"`
	EnforceGovernanceHeader bool   `gorm:"" json:"enforce_governance_header"`
	AllowDirectKeys         bool   `gorm:"" json:"allow_direct_keys"`
	MaxRequestBodySizeMB    int    `gorm:"default:100" json:"max_request_body_size_mb"`
	// LiteLLM fallback flag
	EnableLiteLLMFallbacks bool `gorm:"column:enable_litellm_fallbacks;default:false" json:"enable_litellm_fallbacks"`

	CreatedAt time.Time `gorm:"index;not null" json:"created_at"`
	UpdatedAt time.Time `gorm:"index;not null" json:"updated_at"`

	// Virtual fields for runtime use (not stored in DB)
	PrometheusLabels []string `gorm:"-" json:"prometheus_labels"`
	AllowedOrigins   []string `gorm:"-" json:"allowed_origins,omitempty"`	
}

// TableName sets the table name for each model
func (TableClientConfig) TableName() string { return "config_client" }

func (cc *TableClientConfig) BeforeSave(tx *gorm.DB) error {
	if cc.PrometheusLabels != nil {
		data, err := json.Marshal(cc.PrometheusLabels)
		if err != nil {
			return err
		}
		cc.PrometheusLabelsJSON = string(data)
	} else {
		cc.PrometheusLabelsJSON = "[]"
	}

	if cc.AllowedOrigins != nil {
		data, err := json.Marshal(cc.AllowedOrigins)
		if err != nil {
			return err
		}
		cc.AllowedOriginsJSON = string(data)
	} else {
		cc.AllowedOriginsJSON = "[]"
	}

	return nil
}

// AfterFind hooks for deserialization
func (cc *TableClientConfig) AfterFind(tx *gorm.DB) error {
	if cc.PrometheusLabelsJSON != "" {
		if err := json.Unmarshal([]byte(cc.PrometheusLabelsJSON), &cc.PrometheusLabels); err != nil {
			return err
		}
	}

	if cc.AllowedOriginsJSON != "" {
		if err := json.Unmarshal([]byte(cc.AllowedOriginsJSON), &cc.AllowedOrigins); err != nil {
			return err
		}
	}

	return nil
}
